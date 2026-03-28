use crate::services::{AppState, RequestContext, UsageBreakdown, spawn_usage_recording};
use crate::types::{GeminiGenerateContentResponse, ResponsesRequest};
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use futures::StreamExt;
use std::sync::Arc;
use std::time::Instant;

// Duplicated from adapter/gemini.rs so this module is self-contained.
#[derive(Debug, Default)]
struct SseFrameParser {
    buffer: String,
    data_lines: Vec<String>,
}

impl SseFrameParser {
    fn push_chunk(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut frames = Vec::new();
        self.buffer.push_str(&String::from_utf8_lossy(chunk));

        while let Some(newline_pos) = self.buffer.find('\n') {
            let mut line = self.buffer[..newline_pos].to_string();
            self.buffer.drain(..=newline_pos);

            if line.ends_with('\r') {
                line.pop();
            }

            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    frames.push(self.data_lines.drain(..).collect::<Vec<_>>().join("\n"));
                }
                continue;
            }

            if let Some(rest) = line.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }

        frames
    }

    fn finish(&mut self) -> Vec<String> {
        let mut frames = Vec::new();

        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            if let Some(rest) = line.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }

        if !self.data_lines.is_empty() {
            frames.push(self.data_lines.drain(..).collect::<Vec<_>>().join("\n"));
        }

        frames
    }
}

/// Mutable state threaded through chunk processing.
struct GeminiResponsesStreamState {
    response_id: String,
    created_at: i64,
    sequence_number: u32,
    current_output_index: usize,
    output_items: Vec<serde_json::Value>,

    // Message output item
    message_output_index: Option<usize>,
    message_item_id: Option<String>,
    accumulated_text: String,
    message_content_index: usize,

    // Reasoning output item
    reasoning_output_index: Option<usize>,
    reasoning_item_id: Option<String>,
    accumulated_reasoning: String,

    // Active function calls: (output_index, item_id, call_id, name, accumulated_args)
    active_function_calls: Vec<(usize, String, String, String, String)>,

    // Final usage
    final_input_tokens: u32,
    final_output_tokens: u32,
    final_total_tokens: u32,
}

type SseEventPair = (String, serde_json::Value);

impl GeminiResponsesStreamState {
    fn new(response_id: String, created_at: i64) -> Self {
        Self {
            response_id,
            created_at,
            sequence_number: 0,
            current_output_index: 0,
            output_items: Vec::new(),
            message_output_index: None,
            message_item_id: None,
            accumulated_text: String::new(),
            message_content_index: 0,
            reasoning_output_index: None,
            reasoning_item_id: None,
            accumulated_reasoning: String::new(),
            active_function_calls: Vec::new(),
            final_input_tokens: 0,
            final_output_tokens: 0,
            final_total_tokens: 0,
        }
    }

    /// Convert one Gemini chunk into a list of (event_name, payload) pairs.
    fn process_chunk(&mut self, chunk: GeminiGenerateContentResponse) -> Vec<SseEventPair> {
        let mut events: Vec<SseEventPair> = Vec::new();

        // Accumulate usage metadata from every chunk (last one wins).
        if let Some(meta) = &chunk.usage_metadata {
            if let Some(v) = meta.prompt_token_count {
                self.final_input_tokens = v;
            }
            if let Some(v) = meta.candidates_token_count {
                self.final_output_tokens = v;
            }
            if let Some(v) = meta.total_token_count {
                self.final_total_tokens = v;
            }
        }

        let candidates = match chunk.candidates {
            Some(c) => c,
            None => return events,
        };
        let candidate = match candidates.into_iter().next() {
            Some(c) => c,
            None => return events,
        };
        let content = match candidate.content {
            Some(c) => c,
            None => return events,
        };

        for part in content.parts {
            if part.thought == Some(true) {
                // ── Reasoning / thought part ───────────────────────────────
                let text = match part.text {
                    Some(t) => t,
                    None => continue,
                };

                // Start reasoning output item on first thought chunk.
                if self.reasoning_output_index.is_none() {
                    let item_id = format!("rs_{}", uuid::Uuid::new_v4());
                    let idx = self.current_output_index;
                    let item = serde_json::json!({
                        "type": "reasoning",
                        "id": item_id,
                        "summary": [],
                    });
                    self.output_items.push(item.clone());

                    events.push((
                        "response.output_item.added".to_string(),
                        serde_json::json!({
                            "type": "response.output_item.added",
                            "response_id": self.response_id,
                            "output_index": idx,
                            "item": item,
                            "sequence_number": self.sequence_number
                        }),
                    ));
                    self.sequence_number += 1;

                    events.push((
                        "response.reasoning_summary_part.added".to_string(),
                        serde_json::json!({
                            "type": "response.reasoning_summary_part.added",
                            "response_id": self.response_id,
                            "item_id": item_id,
                            "output_index": idx,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                            "sequence_number": self.sequence_number
                        }),
                    ));
                    self.sequence_number += 1;

                    self.reasoning_item_id = Some(item_id);
                    self.reasoning_output_index = Some(idx);
                    self.current_output_index += 1;
                }

                let output_index = self.reasoning_output_index.unwrap();
                let item_id = self.reasoning_item_id.clone().unwrap_or_default();
                self.accumulated_reasoning.push_str(&text);

                events.push((
                    "response.reasoning_summary_text.delta".to_string(),
                    serde_json::json!({
                        "type": "response.reasoning_summary_text.delta",
                        "response_id": self.response_id,
                        "item_id": item_id,
                        "output_index": output_index,
                        "summary_index": 0,
                        "delta": text,
                        "sequence_number": self.sequence_number
                    }),
                ));
                self.sequence_number += 1;
            } else if let Some(fc) = part.function_call {
                // ── Function call part ────────────────────────────────────
                let fc_id = fc
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                let idx = self.current_output_index;
                let args_str = fc.args.to_string();

                let item = serde_json::json!({
                    "type": "function_call",
                    "id": fc_id,
                    "status": "in_progress",
                    "call_id": fc_id,
                    "name": fc.name,
                    "arguments": "",
                    "thought_signature": part.thought_signature,
                });
                self.output_items.push(item.clone());

                events.push((
                    "response.output_item.added".to_string(),
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "response_id": self.response_id,
                        "output_index": idx,
                        "item": item,
                        "sequence_number": self.sequence_number
                    }),
                ));
                self.sequence_number += 1;

                events.push((
                    "response.function_call_arguments.delta".to_string(),
                    serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "response_id": self.response_id,
                        "delta": args_str,
                        "item_id": fc_id,
                        "output_index": idx,
                        "sequence_number": self.sequence_number
                    }),
                ));
                self.sequence_number += 1;

                self.active_function_calls
                    .push((idx, fc_id.clone(), fc_id, fc.name, args_str));
                self.current_output_index += 1;
            } else if let Some(text) = part.text {
                // ── Regular text part ─────────────────────────────────────
                if self.message_output_index.is_none() {
                    let item_id = format!("msg_{}", uuid::Uuid::new_v4());
                    let idx = self.current_output_index;
                    let item = serde_json::json!({
                        "type": "message",
                        "id": item_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    });
                    self.output_items.push(item.clone());

                    events.push((
                        "response.output_item.added".to_string(),
                        serde_json::json!({
                            "type": "response.output_item.added",
                            "response_id": self.response_id,
                            "output_index": idx,
                            "item": item,
                            "sequence_number": self.sequence_number
                        }),
                    ));
                    self.sequence_number += 1;

                    events.push((
                        "response.content_part.added".to_string(),
                        serde_json::json!({
                            "type": "response.content_part.added",
                            "response_id": self.response_id,
                            "item_id": item_id,
                            "output_index": idx,
                            "content_index": self.message_content_index,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                            "sequence_number": self.sequence_number
                        }),
                    ));
                    self.sequence_number += 1;

                    self.message_item_id = Some(item_id);
                    self.message_output_index = Some(idx);
                    self.current_output_index += 1;
                }

                let output_index = self.message_output_index.unwrap();
                let item_id = self.message_item_id.clone().unwrap_or_default();
                let content_index = self.message_content_index;
                self.accumulated_text.push_str(&text);

                events.push((
                    "response.output_text.delta".to_string(),
                    serde_json::json!({
                        "type": "response.output_text.delta",
                        "response_id": self.response_id,
                        "content_index": content_index,
                        "delta": text,
                        "item_id": item_id,
                        "logprobs": [],
                        "output_index": output_index,
                        "sequence_number": self.sequence_number
                    }),
                ));
                self.sequence_number += 1;
            }
        }

        events
    }

    /// Produce closing events after the byte stream is exhausted.
    fn closing_events(
        &mut self,
        instructions: Option<&String>,
        transport_model_id: &str,
    ) -> Vec<SseEventPair> {
        let mut events: Vec<SseEventPair> = Vec::new();

        // ── Function calls done ────────────────────────────────────────────
        let fcs = self.active_function_calls.clone();
        for (output_index, _item_id, _call_id, _name, accumulated_args) in fcs {
            self.output_items[output_index]["arguments"] = serde_json::json!(accumulated_args);
            self.output_items[output_index]["status"] = serde_json::json!("completed");
            let item = self.output_items[output_index].clone();

            events.push((
                "response.function_call_arguments.done".to_string(),
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": self.response_id,
                    "output_index": output_index,
                    "item": item,
                    "sequence_number": self.sequence_number
                }),
            ));
            self.sequence_number += 1;

            let item = self.output_items[output_index].clone();
            events.push((
                "response.output_item.done".to_string(),
                serde_json::json!({
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "item": item,
                    "output_index": output_index,
                    "sequence_number": self.sequence_number
                }),
            ));
            self.sequence_number += 1;
        }

        // ── Message item done ──────────────────────────────────────────────
        if let Some(output_index) = self.message_output_index {
            let item_id = self.message_item_id.clone().unwrap_or_default();
            let text = self.accumulated_text.clone();
            let content_index = self.message_content_index;

            // Materialise the content part into the stored item.
            if let Some(content) = self.output_items[output_index]
                .get_mut("content")
                .and_then(|v| v.as_array_mut())
            {
                content.push(serde_json::json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }));
            }

            events.push((
                "response.output_text.done".to_string(),
                serde_json::json!({
                    "type": "response.output_text.done",
                    "response_id": self.response_id,
                    "content_index": content_index,
                    "item_id": item_id,
                    "logprobs": [],
                    "output_index": output_index,
                    "sequence_number": self.sequence_number,
                    "text": text
                }),
            ));
            self.sequence_number += 1;

            events.push((
                "response.content_part.done".to_string(),
                serde_json::json!({
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "content_index": content_index,
                    "item_id": item_id,
                    "output_index": output_index,
                    "part": {
                        "type": "output_text",
                        "annotations": [],
                        "text": text
                    },
                    "sequence_number": self.sequence_number
                }),
            ));
            self.sequence_number += 1;

            self.output_items[output_index]["status"] = serde_json::json!("completed");
            let item = self.output_items[output_index].clone();
            events.push((
                "response.output_item.done".to_string(),
                serde_json::json!({
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "item": item,
                    "output_index": output_index,
                    "sequence_number": self.sequence_number
                }),
            ));
            self.sequence_number += 1;
        }

        // ── Reasoning item done ────────────────────────────────────────────
        if let Some(output_index) = self.reasoning_output_index {
            let item_id = self.reasoning_item_id.clone().unwrap_or_default();
            let reasoning_text = self.accumulated_reasoning.clone();

            if let Some(summary) = self.output_items[output_index]
                .get_mut("summary")
                .and_then(|v| v.as_array_mut())
            {
                summary.push(serde_json::json!({
                    "type": "summary_text",
                    "text": reasoning_text
                }));
            }

            events.push((
                "response.reasoning_summary_text.done".to_string(),
                serde_json::json!({
                    "type": "response.reasoning_summary_text.done",
                    "response_id": self.response_id,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": reasoning_text,
                    "sequence_number": self.sequence_number
                }),
            ));
            self.sequence_number += 1;

            events.push((
                "response.reasoning_summary_part.done".to_string(),
                serde_json::json!({
                    "type": "response.reasoning_summary_part.done",
                    "response_id": self.response_id,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": reasoning_text},
                    "sequence_number": self.sequence_number
                }),
            ));
            self.sequence_number += 1;

            let item = self.output_items[output_index].clone();
            events.push((
                "response.output_item.done".to_string(),
                serde_json::json!({
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "item": item,
                    "output_index": output_index,
                    "sequence_number": self.sequence_number
                }),
            ));
            self.sequence_number += 1;
        }

        // ── response.completed ─────────────────────────────────────────────
        let output_items = self.output_items.clone();
        events.push((
            "response.completed".to_string(),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created_at,
                    "status": "completed",
                    "background": false,
                    "completed_at": chrono::Utc::now().timestamp(),
                    "error": null,
                    "instructions": instructions,
                    "model": transport_model_id,
                    "output": output_items,
                    "usage": {
                        "input_tokens": self.final_input_tokens,
                        "output_tokens": self.final_output_tokens,
                        "total_tokens": self.final_total_tokens,
                    }
                },
                "sequence_number": self.sequence_number
            }),
        ));

        events
    }
}

pub(crate) fn gemini_stream_to_responses_sse(
    state: Arc<AppState>,
    ctx: RequestContext,
    request: ResponsesRequest,
    start_time: Instant,
    response: reqwest::Response,
) -> Response {
    let sse_stream = async_stream::stream! {
        let response_id = format!("resp_{}", uuid::Uuid::new_v4());
        let created_at = chrono::Utc::now().timestamp();
        let mut ttft: Option<i32> = None;
        let mut stream_state = GeminiResponsesStreamState::new(response_id.clone(), created_at);

        // ── response.created ──────────────────────────────────────────────
        let initial_response = serde_json::json!({
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "background": false,
            "completed_at": null,
            "error": null,
            "instructions": request.instructions,
            "model": ctx.transport_model_id,
            "output": [],
            "tools": request.tools,
            "temperature": request.temperature.unwrap_or(1.0),
            "max_output_tokens": request.max_output_tokens,
        });

        yield Ok::<_, axum::Error>(Event::default()
            .event("response.created")
            .json_data(serde_json::json!({
                "type": "response.created",
                "response": initial_response,
                "sequence_number": stream_state.sequence_number
            }))
            .unwrap());
        stream_state.sequence_number += 1;

        // ── response.in_progress ──────────────────────────────────────────
        yield Ok::<_, axum::Error>(Event::default()
            .event("response.in_progress")
            .json_data(serde_json::json!({
                "type": "response.in_progress",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                },
                "sequence_number": stream_state.sequence_number
            }))
            .unwrap());
        stream_state.sequence_number += 1;

        // ── Stream processing ─────────────────────────────────────────────
        let mut byte_stream = response.bytes_stream();
        let mut parser = SseFrameParser::default();

        while let Some(chunk_result) = byte_stream.next().await {
            if ttft.is_none() {
                ttft = Some(start_time.elapsed().as_millis() as i32);
            }
            match chunk_result {
                Ok(bytes) => {
                    for frame in parser.push_chunk(&bytes) {
                        let data = frame.trim().to_string();
                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<GeminiGenerateContentResponse>(&data) {
                            Ok(chunk) => {
                                let pairs = stream_state.process_chunk(chunk);
                                for (name, payload) in pairs {
                                    yield Ok::<_, axum::Error>(Event::default()
                                        .event(name)
                                        .json_data(payload)
                                        .unwrap());
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to parse Gemini SSE frame: {}: {}",
                                    e,
                                    data
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Gemini stream error: {:?}", e);
                    yield Err(axum::Error::new(e));
                }
            }
        }

        // Drain any remaining buffered data.
        for frame in parser.finish() {
            let data = frame.trim().to_string();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(chunk) = serde_json::from_str::<GeminiGenerateContentResponse>(&data) {
                let pairs = stream_state.process_chunk(chunk);
                for (name, payload) in pairs {
                    yield Ok::<_, axum::Error>(Event::default()
                        .event(name)
                        .json_data(payload)
                        .unwrap());
                }
            }
        }

        // ── Closing events ────────────────────────────────────────────────
        let closing = stream_state.closing_events(
            request.instructions.as_ref(),
            &ctx.transport_model_id,
        );
        for (name, payload) in closing {
            yield Ok::<_, axum::Error>(Event::default()
                .event(name)
                .json_data(payload)
                .unwrap());
        }

        // Record billing.
        spawn_usage_recording(
            state.clone(),
            ctx.clone(),
            UsageBreakdown {
                input_tokens: stream_state.final_input_tokens as i32,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                reasoning_tokens: 0,
                output_tokens: stream_state.final_output_tokens as i32,
            },
            ttft,
            Some(start_time.elapsed().as_millis() as i32),
        );

        yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));
    };

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
