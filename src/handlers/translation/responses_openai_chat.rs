//! Translations between OpenAI Responses format and OpenAI ChatCompletion format.

use crate::types::{
    ResponseInputItem, ResponseMessageInputContent, ResponseOutputContent, ResponseOutputItem,
    ResponseUsage, ResponsesRequest, ResponsesResponse,
};
use async_openai::types::chat::CreateChatCompletionRequest;
use serde_json::json;

// ─── Public: Responses request → OpenAI Chat request ─────────────────────────

/// OpenAI Responses request → OpenAI ChatCompletion request.
///
/// Uses a serde_json round-trip to avoid constructing all async_openai builder
/// types by hand.
pub(crate) fn convert_responses_to_openai_chat(
    req: &ResponsesRequest,
) -> CreateChatCompletionRequest {
    let mut messages_json: Vec<serde_json::Value> = Vec::new();

    // System instruction → system message (prepended)
    if let Some(instructions) = &req.instructions {
        if !instructions.is_empty() {
            messages_json.push(json!({
                "role": "system",
                "content": instructions
            }));
        }
    }

    // Parse the input array into ChatCompletion messages.
    //
    // Responses API input item types:
    //   Message { role, content }          → user / assistant message
    //   FunctionCall { call_id, name, … }  → assistant message with tool_calls
    //   FunctionCallOutput { call_id, … }  → tool message
    //   Reasoning { … }                    → skip (no equivalent)
    //
    // OpenAI Chat requires that all tool_calls for a single assistant turn be
    // part of the *same* assistant message object.  We therefore buffer
    // consecutive FunctionCall items and flush them as a single assistant
    // message when we encounter anything else.
    #[derive(Default)]
    struct AssistantBuffer {
        text: Option<String>,
        tool_calls: Vec<serde_json::Value>,
    }

    fn flush_assistant(buf: &mut AssistantBuffer, messages: &mut Vec<serde_json::Value>) {
        if buf.text.is_none() && buf.tool_calls.is_empty() {
            return;
        }
        let mut msg = json!({ "role": "assistant" });
        if let Some(text) = buf.text.take() {
            msg["content"] = json!(text);
        }
        if !buf.tool_calls.is_empty() {
            msg["tool_calls"] = json!(std::mem::take(&mut buf.tool_calls));
        }
        messages.push(msg);
        *buf = AssistantBuffer::default();
    }

    let input_items: Vec<ResponseInputItem> = req
        .input
        .as_ref()
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect()
        })
        .unwrap_or_default();

    let mut asst_buf = AssistantBuffer::default();

    for item in input_items {
        match item {
            ResponseInputItem::Message { role, content } => {
                let role_str = role.as_str();
                if role_str == "assistant" {
                    // Buffer assistant text; a subsequent FunctionCall item may add
                    // tool_calls to the same message.
                    let text = match content {
                        ResponseMessageInputContent::String(s) => s,
                        ResponseMessageInputContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| {
                                serde_json::to_value(b).ok().and_then(|v| {
                                    v.get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|s| s.to_string())
                                })
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    };
                    if asst_buf.text.is_none() && asst_buf.tool_calls.is_empty() {
                        asst_buf.text = if text.is_empty() { None } else { Some(text) };
                    } else {
                        // Flush the previous buffered assistant turn first, then start a new one.
                        flush_assistant(&mut asst_buf, &mut messages_json);
                        asst_buf.text = if text.is_empty() { None } else { Some(text) };
                    }
                } else {
                    // Non-assistant message: flush any buffered assistant turn first.
                    flush_assistant(&mut asst_buf, &mut messages_json);
                    let content_str = match content {
                        ResponseMessageInputContent::String(s) => s,
                        ResponseMessageInputContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| {
                                serde_json::to_value(b).ok().and_then(|v| {
                                    v.get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|s| s.to_string())
                                })
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    };
                    let openai_role = if role_str == "developer" {
                        "user"
                    } else {
                        role_str
                    };
                    messages_json.push(json!({
                        "role": openai_role,
                        "content": content_str
                    }));
                }
            }

            ResponseInputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
                ..
            } => {
                // Attach to the current assistant buffer as a tool_call entry.
                let tc_id = id.unwrap_or_else(|| call_id.clone());
                asst_buf.tool_calls.push(json!({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }));
            }

            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                // Flush any buffered assistant turn before the tool result.
                flush_assistant(&mut asst_buf, &mut messages_json);
                let content_str = match output {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                };
                messages_json.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content_str
                }));
            }

            ResponseInputItem::Reasoning { .. } => {
                // No equivalent in OpenAI Chat; skip.
            }
        }
    }

    // Flush any remaining buffered assistant turn.
    flush_assistant(&mut asst_buf, &mut messages_json);

    // Tools
    let tools_json: Option<serde_json::Value> = req.tools.as_ref().and_then(|tools| {
        if tools.is_empty() {
            return None;
        }
        let list: Vec<serde_json::Value> = tools
            .iter()
            .filter_map(|t| {
                let name = t.name.as_ref()?.clone();
                Some(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "strict": t.strict
                    }
                }))
            })
            .collect();
        if list.is_empty() {
            None
        } else {
            Some(json!(list))
        }
    });

    // Tool choice
    let tool_choice_json: Option<serde_json::Value> = req
        .tool_choice
        .as_ref()
        .map(|tc| serde_json::to_value(tc).unwrap_or(json!("auto")));

    let mut request_json = json!({
        "model": req.model,
        "messages": messages_json,
        "stream": req.stream.unwrap_or(false),
    });

    if let Some(max_tokens) = req.max_output_tokens {
        request_json["max_completion_tokens"] = json!(max_tokens);
    }
    if let Some(temp) = req.temperature {
        request_json["temperature"] = json!(temp);
    }
    if let Some(tools) = tools_json {
        request_json["tools"] = tools;
    }
    if let Some(tc) = tool_choice_json {
        request_json["tool_choice"] = tc;
    }

    serde_json::from_value(request_json).unwrap_or_else(|_| {
        serde_json::from_value(json!({
            "model": "unknown",
            "messages": [],
        }))
        .expect("fallback CreateChatCompletionRequest must deserialise")
    })
}

// ─── Public: OpenAI Chat response → Responses response ───────────────────────

/// OpenAI ChatCompletion response JSON → OpenAI Responses response.
pub(crate) fn convert_openai_chat_to_responses(response: &serde_json::Value) -> ResponsesResponse {
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let created_at = response
        .get("created")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp());

    let mut output: Vec<ResponseOutputItem> = Vec::new();

    let message = response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|choice| choice.get("message"));

    if let Some(msg) = message {
        // Text content → Message output item
        if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                output.push(ResponseOutputItem::Message {
                    id: format!("msg_{}", uuid::Uuid::new_v4()),
                    status: Some("completed".to_string()),
                    role: "assistant".to_string(),
                    content: vec![ResponseOutputContent::OutputText {
                        text: text.to_string(),
                        annotations: Some(vec![]),
                    }],
                });
            }
        }

        // Tool calls → FunctionCall output items
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}")
                    .to_string();

                output.push(ResponseOutputItem::FunctionCall {
                    id: id.clone(),
                    status: Some("completed".to_string()),
                    call_id: id,
                    name,
                    arguments,
                    thought_signature: None,
                });
            }
        }
    }

    // Usage
    let usage = response.get("usage").map(|u| ResponseUsage {
        input_tokens: u
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        output_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        total_tokens: u
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    });

    ResponsesResponse {
        id: format!("resp_{}", uuid::Uuid::new_v4()),
        object: "response".to_string(),
        model,
        created_at,
        output: Some(output),
        usage,
    }
}

// ─── Public: OpenAI Chat request → Responses request ─────────────────────────

/// OpenAI ChatCompletion request → OpenAI Responses request.
///
/// Extracts the system message as `instructions`, maps the rest of the messages
/// to the Responses `input` array, and carries over the common scalar fields.
pub(crate) fn convert_chat_request_to_responses(
    req: &CreateChatCompletionRequest,
) -> ResponsesRequest {
    // Serialize to JSON so we can iterate messages without touching every
    // async-openai enum variant by hand.
    let req_json = serde_json::to_value(req).unwrap_or_default();

    let messages = req_json
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    let mut instructions: Option<String> = None;
    let mut input_items: Vec<serde_json::Value> = Vec::new();

    for msg in &messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        match role {
            "system" => {
                // Concatenate all system messages into instructions.
                let text = msg
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                instructions = Some(match instructions.take() {
                    Some(existing) => format!("{}\n{}", existing, text),
                    None => text,
                });
            }
            "assistant" => {
                // Check for tool_calls first.
                if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for tc in tool_calls {
                        let call_id = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}")
                            .to_string();
                        input_items.push(json!({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }));
                    }
                }
                // Text content (may coexist with tool_calls).
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        input_items.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }));
                    }
                }
            }
            "tool" => {
                let call_id = msg
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let output = msg
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                input_items.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));
            }
            _ => {
                // "user" and anything else: forward as-is with content normalised to a string.
                let content_value = msg.get("content").cloned().unwrap_or(json!(""));
                let content_str = match &content_value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                input_items.push(json!({
                    "type": "message",
                    "role": role,
                    "content": content_str,
                }));
            }
        }
    }

    let max_output_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .map(|v| v as u32);

    ResponsesRequest {
        model: req.model.clone(),
        input: Some(json!(input_items)),
        previous_response_id: None,
        instructions,
        stream: req.stream,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        temperature: req.temperature,
        max_output_tokens,
        metadata: None,
    }
}

// ─── Public: Responses response JSON → OpenAI Chat completion response JSON ───

/// OpenAI Responses response JSON → OpenAI ChatCompletion response JSON.
///
/// Useful when the backend spoke `/v1/responses` but the client expects the
/// `/v1/chat/completions` wire format.
pub(crate) fn convert_responses_response_to_openai_chat(
    resp: &serde_json::Value,
) -> serde_json::Value {
    let id = resp
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-unknown")
        .replace("resp_", "chatcmpl-");

    let model = resp
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let created = resp
        .get("created_at")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp());

    let mut text_content: Option<String> = None;
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

    if let Some(output) = resp.get("output").and_then(|o| o.as_array()) {
        for item in output {
            let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match item_type {
                "message" => {
                    if let Some(content_arr) = item.get("content").and_then(|c| c.as_array()) {
                        let parts: Vec<&str> = content_arr
                            .iter()
                            .filter_map(|c| {
                                if c.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                                    c.get("text").and_then(|t| t.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        if !parts.is_empty() {
                            text_content = Some(parts.join(""));
                        }
                    }
                }
                "function_call" => {
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    tool_calls.push(json!({
                        "id": call_id,
                        "type": "function",
                        "function": { "name": name, "arguments": arguments },
                    }));
                }
                _ => {}
            }
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };

    let mut message = json!({ "role": "assistant" });
    if let Some(text) = &text_content {
        message["content"] = json!(text);
    } else {
        message["content"] = json!(null);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }

    let usage = resp.get("usage").map(|u| {
        json!({
            "prompt_tokens": u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "completion_tokens": u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "total_tokens": u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        })
    });

    let mut result = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
    });

    if let Some(u) = usage {
        result["usage"] = u;
    }

    result
}
