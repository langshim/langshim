//! Streaming conversions for the OpenAI Chat ↔ Gemini SSE paths.
//!
//! * `gemini_sse_to_openai_chat_sse` – Gemini SSE in, OpenAI Chat SSE out.
//! * `openai_chat_sse_to_gemini_sse` – OpenAI Chat SSE in, Gemini SSE out.

use crate::services::{AppState, RequestContext, UsageBreakdown, spawn_usage_recording};
use crate::types::{
    GeminiCandidate, GeminiContent, GeminiFunctionCall, GeminiGenerateContentResponse, GeminiPart,
    GeminiUsageMetadata,
};
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use futures::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

// ─── SSE frame parser (duplicated from adapter/gemini.rs) ────────────────────

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

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn map_gemini_finish_to_openai(reason: &str, has_function_calls: bool) -> &'static str {
    if has_function_calls {
        return "tool_calls";
    }
    match reason {
        "MAX_TOKENS" => "length",
        _ => "stop",
    }
}

fn map_openai_finish_to_gemini(reason: &str) -> &'static str {
    match reason {
        "length" => "MAX_TOKENS",
        _ => "STOP",
    }
}

fn make_empty_gemini_part() -> GeminiPart {
    GeminiPart {
        text: None,
        thought: None,
        thought_signature: None,
        inline_data: None,
        file_data: None,
        function_call: None,
        function_response: None,
        cache_control: None,
    }
}

fn make_gemini_text_chunk(model: &str, text: &str) -> GeminiGenerateContentResponse {
    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: vec![GeminiPart {
                    text: Some(text.to_string()),
                    ..make_empty_gemini_part()
                }],
            }),
            finish_reason: None,
            finish_message: None,
            index: Some(0),
        }]),
        usage_metadata: None,
        model_version: Some(model.to_string()),
        response_id: None,
    }
}

fn make_gemini_finish_chunk(
    model: &str,
    parts: Vec<GeminiPart>,
    finish_reason: String,
) -> GeminiGenerateContentResponse {
    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: if parts.is_empty() {
                None
            } else {
                Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts,
                })
            },
            finish_reason: Some(finish_reason),
            finish_message: None,
            index: Some(0),
        }]),
        usage_metadata: None,
        model_version: Some(model.to_string()),
        response_id: None,
    }
}

fn make_gemini_usage_chunk(model: &str, usage: &UsageBreakdown) -> GeminiGenerateContentResponse {
    GeminiGenerateContentResponse {
        candidates: None,
        usage_metadata: Some(GeminiUsageMetadata {
            prompt_token_count: Some(usage.input_tokens.max(0) as u32),
            cached_content_token_count: Some(usage.cache_read_tokens.max(0) as u32),
            candidates_token_count: Some(usage.output_tokens.max(0) as u32),
            total_token_count: Some((usage.input_tokens + usage.output_tokens).max(0) as u32),
        }),
        model_version: Some(model.to_string()),
        response_id: None,
    }
}

/// Drain accumulated tool-call state into a list of Gemini function-call parts.
/// Elements: `(index, id, name, accumulated_args)`.
fn drain_tool_calls(pending: &mut Vec<(usize, String, String, String)>) -> Vec<GeminiPart> {
    if pending.is_empty() {
        return vec![];
    }
    pending.sort_by_key(|(idx, _, _, _)| *idx);
    pending
        .drain(..)
        .map(|(_, id, name, args)| {
            let args_val: serde_json::Value =
                serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
            GeminiPart {
                function_call: Some(GeminiFunctionCall {
                    name,
                    args: args_val,
                    id: if id.is_empty() { None } else { Some(id) },
                }),
                ..make_empty_gemini_part()
            }
        })
        .collect()
}

/// Convert one parsed Gemini SSE frame into OpenAI `chat.completion.chunk` JSON strings.
///
/// Returns `(events, optional_usage_update, updated_tool_call_index)`.
fn gemini_frame_to_openai_events(
    data: &str,
    chat_id: &str,
    model: &str,
    created: i64,
    mut tc_idx: usize,
) -> (Vec<String>, Option<UsageBreakdown>, usize) {
    let mut events: Vec<String> = Vec::new();

    let chunk = match serde_json::from_str::<GeminiGenerateContentResponse>(data) {
        Ok(c) => c,
        Err(_) => return (events, None, tc_idx),
    };

    // Capture usage_metadata whenever it appears (last chunk usually has it).
    let usage = chunk.usage_metadata.as_ref().map(|meta| UsageBreakdown {
        input_tokens: meta.prompt_token_count.unwrap_or(0) as i32,
        output_tokens: meta.candidates_token_count.unwrap_or(0) as i32,
        cache_read_tokens: meta.cached_content_token_count.unwrap_or(0) as i32,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
    });

    if let Some(candidates) = &chunk.candidates {
        if let Some(candidate) = candidates.first() {
            if let Some(content) = &candidate.content {
                for part in &content.parts {
                    // Skip internal thought/reasoning parts.
                    if part.thought == Some(true) {
                        continue;
                    }

                    if let Some(text) = &part.text {
                        events.push(
                            json!({
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": null
                                }]
                            })
                            .to_string(),
                        );
                    }

                    if let Some(fc) = &part.function_call {
                        let call_id = fc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                        events.push(
                            json!({
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": tc_idx,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": &fc.name,
                                                "arguments": fc.args.to_string()
                                            }
                                        }]
                                    },
                                    "finish_reason": null
                                }]
                            })
                            .to_string(),
                        );
                        tc_idx += 1;
                    }
                }
            }

            // Emit a finish chunk when the candidate carries a finish_reason.
            if let Some(fr) = candidate.finish_reason.as_deref() {
                if !fr.is_empty() {
                    let has_fc = candidate
                        .content
                        .as_ref()
                        .map(|c| c.parts.iter().any(|p| p.function_call.is_some()))
                        .unwrap_or(false);
                    let openai_fr = map_gemini_finish_to_openai(fr, has_fc);
                    events.push(
                        json!({
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": openai_fr
                            }]
                        })
                        .to_string(),
                    );
                }
            }
        }
    }

    (events, usage, tc_idx)
}

// ─── Function 1: Gemini SSE → OpenAI Chat SSE ────────────────────────────────

/// Called when: client sends OpenAI Chat request, proxy forwarded to Gemini, Gemini returned SSE.
/// Converts the Gemini SSE stream into an OpenAI Chat SSE stream for the client.
pub(crate) fn gemini_sse_to_openai_chat_sse(
    state: Arc<AppState>,
    ctx: RequestContext,
    start_time: Instant,
    response: reqwest::Response,
) -> Response {
    let sse_stream = async_stream::stream! {
        let mut byte_stream = response.bytes_stream();
        let mut parser = SseFrameParser::default();

        let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model = ctx.resolved_model.clone();
        let mut usage = UsageBreakdown::default();
        let mut ttft: Option<i32> = None;
        let mut tc_idx: usize = 0;

        while let Some(item) = byte_stream.next().await {
            if ttft.is_none() {
                ttft = Some(start_time.elapsed().as_millis() as i32);
            }
            match item {
                Ok(bytes) => {
                    for frame in parser.push_chunk(&bytes) {
                        let data = frame.trim().to_string();
                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }
                        let (events, frame_usage, new_tc_idx) =
                            gemini_frame_to_openai_events(&data, &chat_id, &model, created, tc_idx);
                        tc_idx = new_tc_idx;
                        if let Some(u) = frame_usage {
                            if u.input_tokens > 0 || u.output_tokens > 0 {
                                usage = u;
                            }
                        }
                        for ev in events {
                            yield Ok::<_, axum::Error>(Event::default().data(ev));
                        }
                    }
                }
                Err(e) => {
                    yield Err(axum::Error::new(e));
                }
            }
        }

        // Flush any bytes that remained in the parser buffer without a trailing newline.
        for frame in parser.finish() {
            let data = frame.trim().to_string();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let (events, frame_usage, new_tc_idx) =
                gemini_frame_to_openai_events(&data, &chat_id, &model, created, tc_idx);
            tc_idx = new_tc_idx;
            if let Some(u) = frame_usage {
                if u.input_tokens > 0 || u.output_tokens > 0 {
                    usage = u;
                }
            }
            for ev in events {
                yield Ok::<_, axum::Error>(Event::default().data(ev));
            }
        }

        // OpenAI Chat SSE protocol requires a final [DONE] sentinel.
        yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));

        spawn_usage_recording(
            state.clone(),
            ctx.clone(),
            usage,
            ttft,
            Some(start_time.elapsed().as_millis() as i32),
        );
    };

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ─── Function 2: OpenAI Chat SSE → Gemini SSE ────────────────────────────────

/// Process one parsed OpenAI `chat.completion.chunk` value.
///
/// Mutates `pending_tc` (tool-call accumulation buffer) and `usage`.
/// Returns `(gemini_responses, did_finish)`.
fn process_openai_delta_chunk(
    chunk: &serde_json::Value,
    pending_tc: &mut Vec<(usize, String, String, String)>,
    model: &str,
    usage: &mut UsageBreakdown,
) -> (Vec<GeminiGenerateContentResponse>, bool) {
    let mut responses: Vec<GeminiGenerateContentResponse> = Vec::new();
    let mut did_finish = false;

    // Capture usage when the client requested stream_options.include_usage.
    if let Some(u_obj) = chunk.get("usage").filter(|v| v.is_object()) {
        let input = u_obj
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;
        let output = u_obj
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;
        if input > 0 || output > 0 {
            *usage = UsageBreakdown {
                input_tokens: input,
                output_tokens: output,
                cache_read_tokens: u_obj
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32,
                cache_creation_tokens: 0,
                reasoning_tokens: 0,
            };
        }
    }

    let is_usage_only_chunk = chunk
        .get("choices")
        .and_then(|c| c.as_array())
        .is_some_and(|choices| choices.is_empty());

    if is_usage_only_chunk && (usage.input_tokens > 0 || usage.output_tokens > 0) {
        responses.push(make_gemini_usage_chunk(model, usage));
    }

    let choices = match chunk.get("choices").and_then(|c| c.as_array()) {
        Some(c) => c,
        None => return (responses, did_finish),
    };

    let choice = match choices.first() {
        Some(c) => c,
        None => return (responses, did_finish),
    };

    let finish_reason: Option<String> = choice.get("finish_reason").and_then(|v| {
        if v.is_null() {
            None
        } else {
            v.as_str().map(|s| s.to_string())
        }
    });

    if let Some(delta) = choice.get("delta") {
        // Text content → emit a Gemini text chunk immediately.
        if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                responses.push(make_gemini_text_chunk(model, text));
            }
        }

        // Tool call deltas → accumulate into `pending_tc`.
        if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
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
                    .unwrap_or("");

                if let Some(entry) = pending_tc.iter_mut().find(|(i, _, _, _)| *i == idx) {
                    if !id_str.is_empty() && entry.1.is_empty() {
                        entry.1 = id_str;
                    }
                    if !name_str.is_empty() && entry.2.is_empty() {
                        entry.2 = name_str;
                    }
                    entry.3.push_str(args_delta);
                } else {
                    pending_tc.push((idx, id_str, name_str, args_delta.to_string()));
                }
            }
        }
    }

    // Emit the finish frame (with any buffered tool calls) when finish_reason arrives.
    if let Some(fr) = finish_reason {
        let gemini_fr = map_openai_finish_to_gemini(&fr).to_string();
        let parts = drain_tool_calls(pending_tc);
        responses.push(make_gemini_finish_chunk(model, parts, gemini_fr));
        did_finish = true;
    }

    (responses, did_finish)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn usage_only_openai_chunk_becomes_gemini_usage_event() {
        let mut usage = UsageBreakdown::default();
        let mut pending_tc = Vec::new();
        let chunk = json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19
            }
        });

        let (responses, did_finish) =
            process_openai_delta_chunk(&chunk, &mut pending_tc, "gpt-5.4-nano", &mut usage);

        assert!(!did_finish);
        assert_eq!(responses.len(), 1);
        assert!(responses[0].candidates.is_none());
        assert_eq!(
            responses[0]
                .usage_metadata
                .as_ref()
                .and_then(|m| m.prompt_token_count),
            Some(12)
        );
        assert_eq!(
            responses[0]
                .usage_metadata
                .as_ref()
                .and_then(|m| m.candidates_token_count),
            Some(7)
        );
    }
}

/// Called when: client sends Gemini request, proxy forwarded to OpenAI Chat, OpenAI returned SSE.
/// Converts the OpenAI Chat SSE stream into a Gemini SSE stream for the client.
pub(crate) fn openai_chat_sse_to_gemini_sse(
    state: Arc<AppState>,
    ctx: RequestContext,
    start_time: Instant,
    response: reqwest::Response,
) -> Response {
    let sse_stream = async_stream::stream! {
            let mut byte_stream = response.bytes_stream();
            let mut parser = SseFrameParser::default();

            let model = ctx.resolved_model.clone();
            let mut usage = UsageBreakdown::default();
            let mut ttft: Option<i32> = None;
            // Tool call accumulation: (index, id, name, accumulated_args)
            let mut pending_tc: Vec<(usize, String, String, String)> = Vec::new();
            let mut emitted_finish = false;

            while let Some(item) = byte_stream.next().await {
                if ttft.is_none() {
                    ttft = Some(start_time.elapsed().as_millis() as i32);
                }
                match item {
                    Ok(bytes) => {
                        for frame in parser.push_chunk(&bytes) {
                            let data = frame.trim().to_string();
                            if data.is_empty() || data == "[DONE]" {
                                continue;
                            }
                            let chunk: serde_json::Value = match serde_json::from_str(&data) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };

                            let (gemini_responses, did_finish) =
                                process_openai_delta_chunk(&chunk, &mut pending_tc, &model, &mut usage);

                            for gr in gemini_responses {
                                if let Ok(ev) = Event::default().json_data(&gr) {
                                    yield Ok::<_, axum::Error>(ev);
                                }
                            }

                            if did_finish {
                                emitted_finish = true;
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(axum::Error::new(e));
                    }
                }
            }

            // Flush any remaining bytes from the parser buffer.
            for frame in parser.finish() {
                let data = frame.trim().to_string();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let chunk: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let (gemini_responses, did_finish) =
                    process_openai_delta_chunk(&chunk, &mut pending_tc, &model, &mut usage);

                for gr in gemini_responses {
                    if let Ok(ev) = Event::default().json_data(&gr) {
                        yield Ok::<_, axum::Error>(ev);
                    }
                }

                if did_finish {
                    emitted_finish = true;
                }
            }

            // If no finish frame was emitted (e.g. stream ended abruptly), emit a synthetic one.
            if !emitted_finish {
                let parts = drain_tool_calls(&mut pending_tc);
                let gr = make_gemini_finish_chunk(&model, parts, "STOP".to_string());
                if let Ok(ev) = Event::default().json_data(&gr) {
                    yield Ok::<_, axum::Error>(ev);
                }
            }

            // Gemini SSE protocol does not use a [DONE] sentinel – the stream just closes.

        spawn_usage_recording(
            state.clone(),
            ctx.clone(),
            usage,
            ttft,
            Some(start_time.elapsed().as_millis() as i32),
        );
    };

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
