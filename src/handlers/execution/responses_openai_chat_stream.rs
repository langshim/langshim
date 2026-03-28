//! Streaming conversions between OpenAI Chat SSE and OpenAI Responses SSE formats.
//!
//! * `openai_chat_sse_to_responses_sse` – Chat SSE in, Responses SSE out.
//! * `responses_sse_to_openai_chat_sse` – Responses SSE in, Chat SSE out.

use crate::services::{AppState, RequestContext, UsageBreakdown, spawn_usage_recording};
use crate::types::ResponsesRequest;
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use futures::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

// ─── SSE frame parser (same pattern as other execution modules) ───────────────

#[derive(Debug, Default)]
struct SseFrameParser {
    buffer: String,
    data_lines: Vec<String>,
}

impl SseFrameParser {
    fn push_chunk(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut frames = Vec::new();
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        while let Some(pos) = self.buffer.find('\n') {
            let mut line = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos);
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

// ─── Event pair type ──────────────────────────────────────────────────────────

/// `(event_name, payload)` – yielded to the SSE client.
type SsePair = (String, serde_json::Value);

// ─── Stream state ─────────────────────────────────────────────────────────────

/// All mutable state accumulated while consuming the OpenAI Chat SSE stream.
struct ChatStreamState {
    // Stored so they can be echoed in response.completed / initial events.
    request: ResponsesRequest,
    transport_model_id: String,

    response_id: String,
    created_at: i64,
    sequence_number: u32,

    // Output item tracking
    current_output_index: usize,
    output_items: Vec<serde_json::Value>,

    // Message output item
    message_output_index: Option<usize>,
    message_item_id: Option<String>,
    text_content_started: bool,
    accumulated_text: String,

    // Tool call tracking: (tc_idx, output_index, call_id, name, accumulated_args)
    tool_calls: Vec<(usize, usize, String, String, String)>,

    // Usage (populated only when the client requested include_usage)
    pub input_tokens: i32,
    pub output_tokens: i32,

    /// Set to true once we have emitted response.completed.
    pub completed: bool,
}

impl ChatStreamState {
    fn new(
        response_id: String,
        created_at: i64,
        ctx: &RequestContext,
        request: &ResponsesRequest,
    ) -> Self {
        Self {
            request: request.clone(),
            transport_model_id: ctx.transport_model_id.clone(),
            response_id,
            created_at,
            sequence_number: 0,
            current_output_index: 0,
            output_items: Vec::new(),
            message_output_index: None,
            message_item_id: None,
            text_content_started: false,
            accumulated_text: String::new(),
            tool_calls: Vec::new(),
            input_tokens: 0,
            output_tokens: 0,
            completed: false,
        }
    }

    fn next_seq(&mut self) -> u32 {
        let s = self.sequence_number;
        self.sequence_number += 1;
        s
    }

    // ── Initial "response.created" / "response.in_progress" events ───────────

    fn initial_events(&mut self) -> Vec<SsePair> {
        let initial_response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "background": false,
            "completed_at": null,
            "error": null,
            "instructions": self.request.instructions,
            "model": self.transport_model_id,
            "output": [],
            "tools": self.request.tools,
            "temperature": self.request.temperature.unwrap_or(1.0),
            "max_output_tokens": self.request.max_output_tokens,
        });

        vec![
            (
                "response.created".to_string(),
                json!({
                    "type": "response.created",
                    "response": initial_response,
                    "sequence_number": self.next_seq()
                }),
            ),
            (
                "response.in_progress".to_string(),
                json!({
                    "type": "response.in_progress",
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "status": "in_progress"
                    },
                    "sequence_number": self.next_seq()
                }),
            ),
        ]
    }

    // ── Per-frame processing ─────────────────────────────────────────────────

    /// Process one raw `data:` payload from the OpenAI Chat SSE stream.
    /// Returns zero or more (event_name, payload) pairs to forward to the client.
    fn process_frame(&mut self, data: &str) -> Vec<SsePair> {
        let mut events: Vec<SsePair> = Vec::new();

        if data.is_empty() || data == "[DONE]" {
            return events;
        }

        let chunk: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return events,
        };

        // Capture usage when the caller has set stream_options.include_usage.
        if let Some(u) = chunk.get("usage").filter(|v| v.is_object()) {
            let pt = u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let ct = u
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;
            if pt > 0 {
                self.input_tokens = pt;
            }
            if ct > 0 {
                self.output_tokens = ct;
            }
        }

        let choices = match chunk.get("choices").and_then(|c| c.as_array()) {
            Some(c) if !c.is_empty() => c,
            _ => return events,
        };
        let choice = &choices[0];

        let finish_reason: Option<String> = choice
            .get("finish_reason")
            .filter(|v| !v.is_null())
            .and_then(|v| v.as_str().map(|s| s.to_string()));

        if let Some(delta) = choice.get("delta") {
            // ── Text content ─────────────────────────────────────────────────
            if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                if !text.is_empty() {
                    // First text delta: emit output_item.added + content_part.added
                    if !self.text_content_started {
                        let item_id = format!("msg_{}", uuid::Uuid::new_v4());
                        let output_index = self.current_output_index;
                        self.current_output_index += 1;

                        self.output_items.push(json!({
                            "type": "message",
                            "id": item_id,
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": ""}]
                        }));

                        self.message_item_id = Some(item_id.clone());
                        self.message_output_index = Some(output_index);
                        self.text_content_started = true;

                        let seq = self.next_seq();
                        events.push((
                            "response.output_item.added".to_string(),
                            json!({
                                "type": "response.output_item.added",
                                "response_id": self.response_id,
                                "output_index": output_index,
                                "item": {
                                    "type": "message",
                                    "id": item_id,
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": []
                                },
                                "sequence_number": seq
                            }),
                        ));

                        let seq = self.next_seq();
                        events.push((
                            "response.content_part.added".to_string(),
                            json!({
                                "type": "response.content_part.added",
                                "response_id": self.response_id,
                                "output_index": output_index,
                                "item_id": item_id,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []},
                                "sequence_number": seq
                            }),
                        ));
                    }

                    self.accumulated_text.push_str(text);

                    let item_id = self.message_item_id.clone().unwrap_or_default();
                    let output_index = self.message_output_index.unwrap_or(0);
                    let seq = self.next_seq();
                    events.push((
                        "response.output_text.delta".to_string(),
                        json!({
                            "type": "response.output_text.delta",
                            "response_id": self.response_id,
                            "content_index": 0,
                            "delta": text,
                            "item_id": item_id,
                            "logprobs": [],
                            "output_index": output_index,
                            "sequence_number": seq
                        }),
                    ));
                }
            }

            // ── Tool calls ────────────────────────────────────────────────────
            if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tcs {
                    let tc_idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let id_str = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name_str = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args_delta = tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Find an existing entry for this tc_idx.
                    let existing_pos = self
                        .tool_calls
                        .iter()
                        .position(|(i, _, _, _, _)| *i == tc_idx);

                    if let Some(pos) = existing_pos {
                        // Accumulate arguments delta.
                        self.tool_calls[pos].4.push_str(&args_delta);

                        if !args_delta.is_empty() {
                            let call_id = self.tool_calls[pos].2.clone();
                            let output_index = self.tool_calls[pos].1;
                            let seq = self.next_seq();
                            events.push((
                                "response.function_call_arguments.delta".to_string(),
                                json!({
                                    "type": "response.function_call_arguments.delta",
                                    "response_id": self.response_id,
                                    "delta": args_delta,
                                    "item_id": call_id,
                                    "output_index": output_index,
                                    "sequence_number": seq
                                }),
                            ));
                        }
                    } else {
                        // New tool call: emit output_item.added.
                        let call_id = if !id_str.is_empty() {
                            id_str
                        } else {
                            format!("call_{}", uuid::Uuid::new_v4())
                        };
                        let output_index = self.current_output_index;
                        self.current_output_index += 1;

                        self.output_items.push(json!({
                            "type": "function_call",
                            "id": call_id,
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": name_str,
                            "arguments": "",
                            "thought_signature": null
                        }));

                        let item_snapshot = self.output_items[output_index].clone();
                        let seq = self.next_seq();
                        events.push((
                            "response.output_item.added".to_string(),
                            json!({
                                "type": "response.output_item.added",
                                "response_id": self.response_id,
                                "output_index": output_index,
                                "item": item_snapshot,
                                "sequence_number": seq
                            }),
                        ));

                        // Store the entry before borrowing for the delta event below.
                        self.tool_calls.push((
                            tc_idx,
                            output_index,
                            call_id.clone(),
                            name_str,
                            args_delta.clone(),
                        ));

                        // Emit initial delta if the first chunk already carried arguments.
                        if !args_delta.is_empty() {
                            let seq = self.next_seq();
                            events.push((
                                "response.function_call_arguments.delta".to_string(),
                                json!({
                                    "type": "response.function_call_arguments.delta",
                                    "response_id": self.response_id,
                                    "delta": args_delta,
                                    "item_id": call_id,
                                    "output_index": output_index,
                                    "sequence_number": seq
                                }),
                            ));
                        }
                    }
                }
            }
        }

        // ── Finish reason → emit all closing events ───────────────────────────
        if finish_reason.is_some() {
            events.extend(self.closing_events());
        }

        events
    }

    // ── Closing events (message done + tool calls done + response.completed) ──

    /// Emit all events required to properly close the response.
    /// Safe to call multiple times; subsequent calls return nothing.
    fn closing_events(&mut self) -> Vec<SsePair> {
        if self.completed {
            return vec![];
        }
        self.completed = true;

        let mut events: Vec<SsePair> = Vec::new();

        // ── Close message output item ─────────────────────────────────────────
        if let (Some(item_id), Some(output_index)) =
            (self.message_item_id.clone(), self.message_output_index)
        {
            let full_text = self.accumulated_text.clone();

            // Update the stored item to "completed".
            if let Some(item) = self.output_items.get_mut(output_index) {
                *item = json!({
                    "type": "message",
                    "id": item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": full_text,
                        "annotations": []
                    }]
                });
            }

            let seq = self.next_seq();
            events.push((
                "response.output_text.done".to_string(),
                json!({
                    "type": "response.output_text.done",
                    "response_id": self.response_id,
                    "content_index": 0,
                    "item_id": item_id,
                    "logprobs": [],
                    "output_index": output_index,
                    "sequence_number": seq,
                    "text": full_text
                }),
            ));

            let seq = self.next_seq();
            events.push((
                "response.content_part.done".to_string(),
                json!({
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "content_index": 0,
                    "item_id": item_id,
                    "output_index": output_index,
                    "part": {
                        "type": "output_text",
                        "annotations": [],
                        "text": full_text
                    },
                    "sequence_number": seq
                }),
            ));

            let final_item = self
                .output_items
                .get(output_index)
                .cloned()
                .unwrap_or(json!(null));
            let seq = self.next_seq();
            events.push((
                "response.output_item.done".to_string(),
                json!({
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "item": final_item,
                    "output_index": output_index,
                    "sequence_number": seq
                }),
            ));
        }

        // ── Close tool call output items ──────────────────────────────────────
        let tool_calls_snapshot = self.tool_calls.clone();
        for (_, output_index, call_id, name, args) in tool_calls_snapshot {
            let completed_item = json!({
                "type": "function_call",
                "id": call_id,
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": args
            });

            if let Some(item) = self.output_items.get_mut(output_index) {
                *item = completed_item.clone();
            }

            let seq = self.next_seq();
            events.push((
                "response.function_call_arguments.done".to_string(),
                json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": self.response_id,
                    "output_index": output_index,
                    "item": completed_item,
                    "sequence_number": seq
                }),
            ));

            let final_item = self
                .output_items
                .get(output_index)
                .cloned()
                .unwrap_or(json!(null));
            let seq = self.next_seq();
            events.push((
                "response.output_item.done".to_string(),
                json!({
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "item": final_item,
                    "output_index": output_index,
                    "sequence_number": seq
                }),
            ));
        }

        // ── response.completed ────────────────────────────────────────────────
        let seq = self.next_seq();
        events.push((
            "response.completed".to_string(),
            json!({
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created_at,
                    "status": "completed",
                    "background": false,
                    "completed_at": chrono::Utc::now().timestamp(),
                    "error": null,
                    "instructions": self.request.instructions,
                    "model": self.transport_model_id,
                    "output": self.output_items,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens,
                        "total_tokens": self.input_tokens + self.output_tokens
                    }
                },
                "sequence_number": seq
            }),
        ));

        events
    }
}

// ─── Public streaming function ────────────────────────────────────────────────

/// Called when: client sends Responses request, proxy forwarded to OpenAI Chat,
/// OpenAI returned SSE.  Converts the OpenAI Chat SSE stream into a Responses
/// SSE stream for the client.
// ─── Responses SSE → OpenAI Chat SSE ─────────────────────────────────────────

/// A minimal SSE frame parser that preserves both `event:` and `data:` lines.
#[derive(Debug, Default)]
struct ResponsesSseFrameParser {
    buffer: String,
    current_event: Option<String>,
    data_lines: Vec<String>,
}

impl ResponsesSseFrameParser {
    /// Push a raw byte chunk; returns fully-assembled `(event_name, data)` frames.
    fn push_chunk(&mut self, chunk: &[u8]) -> Vec<(Option<String>, String)> {
        let mut frames = Vec::new();
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        while let Some(pos) = self.buffer.find('\n') {
            let mut line = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos);
            if line.ends_with('\r') {
                line.pop();
            }
            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    let data = self.data_lines.drain(..).collect::<Vec<_>>().join("\n");
                    frames.push((self.current_event.take(), data));
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("event:") {
                self.current_event = Some(rest.trim_start().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }
        frames
    }

    fn finish(&mut self) -> Vec<(Option<String>, String)> {
        let mut frames = Vec::new();
        if !self.data_lines.is_empty() {
            let data = self.data_lines.drain(..).collect::<Vec<_>>().join("\n");
            frames.push((self.current_event.take(), data));
        }
        frames
    }
}

/// Convert a single Responses SSE frame into zero or more OpenAI Chat
/// `chat.completion.chunk` JSON strings.
///
/// Returns `(chunks, Option<UsageBreakdown>)`.
fn responses_frame_to_chat_chunks(
    event_name: Option<&str>,
    data: &str,
    chat_id: &str,
    model: &str,
    created: i64,
    first_chunk_sent: &mut bool,
) -> (Vec<String>, Option<UsageBreakdown>) {
    if data == "[DONE]" {
        return (vec![], None);
    }

    let frame: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return (vec![], None),
    };

    let mut chunks: Vec<String> = Vec::new();

    let make_chunk = |delta: serde_json::Value, finish_reason: Option<&str>| -> String {
        serde_json::to_string(&json!({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }]
        }))
        .unwrap_or_default()
    };

    let event =
        event_name.unwrap_or_else(|| frame.get("type").and_then(|t| t.as_str()).unwrap_or(""));

    match event {
        // First meaningful content event – send role bootstrap chunk.
        "response.output_item.added" | "response.content_part.added" => {
            if !*first_chunk_sent {
                *first_chunk_sent = true;
                chunks.push(make_chunk(
                    json!({"role": "assistant", "content": ""}),
                    None,
                ));
            }
        }

        // Text delta.
        "response.output_text.delta" => {
            if !*first_chunk_sent {
                *first_chunk_sent = true;
                chunks.push(make_chunk(
                    json!({"role": "assistant", "content": ""}),
                    None,
                ));
            }
            if let Some(delta) = frame.get("delta").and_then(|d| d.as_str()) {
                if !delta.is_empty() {
                    chunks.push(make_chunk(json!({"content": delta}), None));
                }
            }
        }

        // Tool call argument delta.
        "response.function_call_arguments.delta" => {
            if !*first_chunk_sent {
                *first_chunk_sent = true;
                chunks.push(make_chunk(
                    json!({"role": "assistant", "content": ""}),
                    None,
                ));
            }
            let index = frame
                .get("output_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let call_id = frame
                .get("item_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let delta = frame.get("delta").and_then(|d| d.as_str()).unwrap_or("");
            chunks.push(make_chunk(
                json!({
                    "tool_calls": [{
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"arguments": delta}
                    }]
                }),
                None,
            ));
        }

        // Stream completed – emit finish + usage.
        "response.completed" => {
            let response = frame.get("response").unwrap_or(&frame);

            // Determine finish reason.
            let has_tool_calls = response
                .get("output")
                .and_then(|o| o.as_array())
                .map(|arr| {
                    arr.iter().any(|item| {
                        item.get("type").and_then(|t| t.as_str()) == Some("function_call")
                    })
                })
                .unwrap_or(false);
            let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };

            chunks.push(make_chunk(json!({}), Some(finish_reason)));

            // Extract usage.
            let usage = response.get("usage").map(|u| UsageBreakdown {
                input_tokens: u.get("input_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
                output_tokens: u.get("output_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                reasoning_tokens: 0,
            });

            return (chunks, usage);
        }

        _ => {}
    }

    (chunks, None)
}

/// Stream `/v1/responses` SSE from the backend, re-emitting each event as an
/// OpenAI `chat.completion.chunk` for clients that speak `/v1/chat/completions`.
pub(crate) fn responses_sse_to_openai_chat_sse(
    state: Arc<AppState>,
    ctx: RequestContext,
    start_time: Instant,
    response: reqwest::Response,
) -> Response {
    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model = ctx.resolved_model.clone();

    let sse_stream = async_stream::stream! {
        let mut byte_stream = response.bytes_stream();
        let mut parser = ResponsesSseFrameParser::default();
        let mut ttft: Option<i32> = None;
        let mut first_chunk_sent = false;
        let mut final_usage: Option<UsageBreakdown> = None;

        while let Some(item) = byte_stream.next().await {
            if ttft.is_none() {
                ttft = Some(start_time.elapsed().as_millis() as i32);
            }
            match item {
                Ok(bytes) => {
                    for (event_name, data) in parser.push_chunk(&bytes) {
                        let (chunk_strs, usage) = responses_frame_to_chat_chunks(
                            event_name.as_deref(),
                            &data,
                            &chat_id,
                            &model,
                            created,
                            &mut first_chunk_sent,
                        );
                        if let Some(u) = usage {
                            final_usage = Some(u);
                        }
                        for chunk in chunk_strs {
                            yield Ok::<_, axum::Error>(Event::default().data(chunk));
                        }
                    }
                }
                Err(e) => {
                    yield Err(axum::Error::new(e));
                }
            }
        }

        for (event_name, data) in parser.finish() {
            let (chunk_strs, usage) = responses_frame_to_chat_chunks(
                event_name.as_deref(),
                &data,
                &chat_id,
                &model,
                created,
                &mut first_chunk_sent,
            );
            if let Some(u) = usage {
                final_usage = Some(u);
            }
            for chunk in chunk_strs {
                yield Ok::<_, axum::Error>(Event::default().data(chunk));
            }
        }

        yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));

        spawn_usage_recording(
            state.clone(),
            ctx.clone(),
            final_usage.unwrap_or_default(),
            ttft,
            Some(start_time.elapsed().as_millis() as i32),
        );
    };

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ─── OpenAI Chat SSE → Responses SSE ─────────────────────────────────────────

pub(crate) fn openai_chat_sse_to_responses_sse(
    state: Arc<AppState>,
    ctx: RequestContext,
    request: ResponsesRequest,
    start_time: Instant,
    response: reqwest::Response,
) -> Response {
    let sse_stream = async_stream::stream! {
        let mut byte_stream = response.bytes_stream();
        let mut parser = SseFrameParser::default();
        let mut ttft: Option<i32> = None;

        let response_id = format!("resp_{}", uuid::Uuid::new_v4());
        let created_at = chrono::Utc::now().timestamp();
        let mut stream_state = ChatStreamState::new(response_id, created_at, &ctx, &request);

        // Emit initial events.
        for (event_name, payload) in stream_state.initial_events() {
            yield Ok::<_, axum::Error>(
                Event::default()
                    .event(event_name)
                    .json_data(payload)
                    .unwrap(),
            );
        }

        // Consume the HTTP response body frame by frame.
        while let Some(item) = byte_stream.next().await {
            if ttft.is_none() {
                ttft = Some(start_time.elapsed().as_millis() as i32);
            }
            match item {
                Ok(bytes) => {
                    for frame in parser.push_chunk(&bytes) {
                        let data = frame.trim().to_string();
                        for (event_name, payload) in stream_state.process_frame(&data) {
                            yield Ok::<_, axum::Error>(
                                Event::default()
                                    .event(event_name)
                                    .json_data(payload)
                                    .unwrap(),
                            );
                        }
                    }
                }
                Err(e) => {
                    yield Err(axum::Error::new(e));
                }
            }
        }

        // Flush any bytes that remained in the buffer without a trailing newline.
        for frame in parser.finish() {
            let data = frame.trim().to_string();
            for (event_name, payload) in stream_state.process_frame(&data) {
                yield Ok::<_, axum::Error>(
                    Event::default()
                        .event(event_name)
                        .json_data(payload)
                        .unwrap(),
                );
            }
        }

        // If the stream ended without a finish_reason chunk, emit closing events now.
        for (event_name, payload) in stream_state.closing_events() {
            yield Ok::<_, axum::Error>(
                Event::default()
                    .event(event_name)
                    .json_data(payload)
                    .unwrap(),
            );
        }

        // Emit [DONE] for clients that expect it (compatible with OpenAI Chat format).
        yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));

        spawn_usage_recording(
            state.clone(),
            ctx.clone(),
            UsageBreakdown {
                input_tokens: stream_state.input_tokens,
                output_tokens: stream_state.output_tokens,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                reasoning_tokens: 0,
            },
            ttft,
            Some(start_time.elapsed().as_millis() as i32),
        );
    };

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
