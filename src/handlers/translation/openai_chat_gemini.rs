//! Translations between OpenAI ChatCompletion format and Gemini generateContent format.

use crate::types::{
    GeminiCandidate, GeminiContent, GeminiFileData, GeminiFunctionCall, GeminiFunctionDeclaration,
    GeminiFunctionResponse, GeminiGenerateContentRequest, GeminiGenerateContentResponse,
    GeminiGenerationConfig, GeminiInlineData, GeminiPart, GeminiTool, GeminiUsageMetadata,
};
use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionStreamOptions, ChatCompletionTools, CreateChatCompletionRequest,
};
use serde_json::json;
use std::collections::{HashMap, VecDeque};

// ─── Private helpers ─────────────────────────────────────────────────────────

fn make_text_part(text: String) -> GeminiPart {
    GeminiPart {
        text: Some(text),
        thought: None,
        thought_signature: None,
        inline_data: None,
        file_data: None,
        function_call: None,
        function_response: None,
        cache_control: None,
    }
}

fn empty_gemini_part() -> GeminiPart {
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

/// Append `parts` to the last GeminiContent when roles match; otherwise push a new content block.
/// This ensures Gemini's alternating-role requirement is met.
fn push_or_merge_contents(contents: &mut Vec<GeminiContent>, role: String, parts: Vec<GeminiPart>) {
    if let Some(last) = contents.last_mut() {
        if last.role.as_deref() == Some(role.as_str()) {
            last.parts.extend(parts);
            return;
        }
    }
    contents.push(GeminiContent {
        role: Some(role),
        parts,
    });
}

fn user_part_to_gemini(part: &ChatCompletionRequestUserMessageContentPart) -> Option<GeminiPart> {
    match part {
        ChatCompletionRequestUserMessageContentPart::Text(text_part) => {
            Some(make_text_part(text_part.text.clone()))
        }
        ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
            let url = image_part.image_url.url.clone();
            if let Some(rest) = url.strip_prefix("data:") {
                if let Some((mime_and_enc, data)) = rest.split_once(',') {
                    let mime_type = mime_and_enc
                        .split(';')
                        .next()
                        .unwrap_or("image/jpeg")
                        .to_string();
                    Some(GeminiPart {
                        inline_data: Some(GeminiInlineData {
                            mime_type,
                            data: data.to_string(),
                        }),
                        ..empty_gemini_part()
                    })
                } else {
                    None
                }
            } else {
                Some(GeminiPart {
                    file_data: Some(GeminiFileData {
                        mime_type: "image/jpeg".to_string(),
                        file_uri: url,
                    }),
                    ..empty_gemini_part()
                })
            }
        }
        _ => None,
    }
}

fn system_content_to_text(content: &ChatCompletionRequestSystemMessageContent) -> String {
    match content {
        ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
        _ => serde_json::to_string(content).unwrap_or_default(),
    }
}

fn assistant_text(content: &ChatCompletionRequestAssistantMessageContent) -> Option<String> {
    match content {
        ChatCompletionRequestAssistantMessageContent::Text(t) if !t.is_empty() => Some(t.clone()),
        _ => None,
    }
}

// ─── Public: OpenAI Chat request → Gemini request ────────────────────────────

/// OpenAI ChatCompletion request → Gemini generateContent request.
pub(crate) fn convert_openai_chat_to_gemini_request(
    req: &CreateChatCompletionRequest,
) -> GeminiGenerateContentRequest {
    let mut system_texts: Vec<String> = Vec::new();
    let mut contents: Vec<GeminiContent> = Vec::new();

    for msg in &req.messages {
        match msg {
            ChatCompletionRequestMessage::System(sys) => {
                let text = system_content_to_text(&sys.content);
                if !text.is_empty() {
                    system_texts.push(text);
                }
            }
            ChatCompletionRequestMessage::Developer(dev) => {
                // Treat developer messages as additional system context.
                let raw = serde_json::to_value(&dev.content)
                    .ok()
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();
                if !raw.is_empty() {
                    system_texts.push(raw);
                }
            }
            ChatCompletionRequestMessage::User(user) => {
                let parts: Vec<GeminiPart> = match &user.content {
                    ChatCompletionRequestUserMessageContent::Text(text) => {
                        vec![make_text_part(text.clone())]
                    }
                    ChatCompletionRequestUserMessageContent::Array(arr) => {
                        arr.iter().filter_map(user_part_to_gemini).collect()
                    }
                };
                if !parts.is_empty() {
                    push_or_merge_contents(&mut contents, "user".to_string(), parts);
                }
            }
            ChatCompletionRequestMessage::Assistant(asst) => {
                let mut parts: Vec<GeminiPart> = Vec::new();

                if let Some(content) = &asst.content {
                    if let Some(text) = assistant_text(content) {
                        parts.push(make_text_part(text));
                    }
                }

                // Tool-call portions → Gemini function_call parts
                if let Some(tool_calls) = &asst.tool_calls {
                    for tc in tool_calls {
                        if let ChatCompletionMessageToolCalls::Function(f) = tc {
                            let args: serde_json::Value =
                                serde_json::from_str(&f.function.arguments)
                                    .unwrap_or_else(|_| json!({}));
                            parts.push(GeminiPart {
                                function_call: Some(GeminiFunctionCall {
                                    name: f.function.name.clone(),
                                    args,
                                    id: Some(f.id.clone()),
                                }),
                                ..empty_gemini_part()
                            });
                        }
                    }
                }

                if !parts.is_empty() {
                    push_or_merge_contents(&mut contents, "model".to_string(), parts);
                }
            }
            ChatCompletionRequestMessage::Tool(tool_msg) => {
                // Serialise content to a String regardless of the concrete enum variant.
                let content_str: String = serde_json::to_value(&tool_msg.content)
                    .ok()
                    .and_then(|v| match v {
                        serde_json::Value::String(s) => Some(s),
                        serde_json::Value::Array(arr) => Some(
                            arr.iter()
                                .filter_map(|item| {
                                    item.get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|s| s.to_string())
                                })
                                .collect::<Vec<_>>()
                                .join(""),
                        ),
                        _ => None,
                    })
                    .unwrap_or_default();

                let response_body: serde_json::Value = serde_json::from_str(&content_str)
                    .unwrap_or_else(|_| json!({ "content": content_str }));

                push_or_merge_contents(
                    &mut contents,
                    "user".to_string(),
                    vec![GeminiPart {
                        function_response: Some(GeminiFunctionResponse {
                            // tool_call_id is the best available name for the Gemini function
                            name: tool_msg.tool_call_id.clone(),
                            response: response_body,
                        }),
                        ..empty_gemini_part()
                    }],
                );
            }
            // Function (legacy) and any future variants — skip
            _ => {}
        }
    }

    let system_instruction = if system_texts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: None,
            parts: vec![make_text_part(system_texts.join("\n\n"))],
        })
    };

    let tools = req.tools.as_ref().and_then(|tools_list| {
        if tools_list.is_empty() {
            return None;
        }
        let declarations: Vec<GeminiFunctionDeclaration> = tools_list
            .iter()
            .filter_map(|t| {
                if let ChatCompletionTools::Function(f) = t {
                    Some(GeminiFunctionDeclaration {
                        name: f.function.name.clone(),
                        description: f.function.description.clone(),
                        parameters: f.function.parameters.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();
        if declarations.is_empty() {
            None
        } else {
            Some(vec![GeminiTool {
                function_declarations: Some(declarations),
            }])
        }
    });

    let max_output_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .map(|t| t as u32);

    let generation_config =
        if max_output_tokens.is_some() || req.temperature.is_some() || req.top_p.is_some() {
            Some(GeminiGenerationConfig {
                max_output_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
                stop_sequences: None,
            })
        } else {
            None
        };

    GeminiGenerateContentRequest {
        contents,
        system_instruction,
        generation_config,
        tools,
    }
}

// ─── Public: Gemini response → OpenAI Chat response ──────────────────────────

/// Gemini generateContent response → OpenAI ChatCompletion response JSON.
pub(crate) fn convert_gemini_response_to_openai_chat(
    model: &str,
    response: GeminiGenerateContentResponse,
) -> serde_json::Value {
    let created = chrono::Utc::now().timestamp();
    let id = response
        .response_id
        .clone()
        .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()));

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut finish_reason = "stop".to_string();

    if let Some(candidates) = &response.candidates {
        if let Some(candidate) = candidates.first() {
            if let Some(fr) = &candidate.finish_reason {
                finish_reason = match fr.as_str() {
                    "STOP" | "STOP_SEQUENCE" => "stop",
                    "MAX_TOKENS" => "length",
                    _ => "stop",
                }
                .to_string();
            }

            if let Some(content) = &candidate.content {
                for part in &content.parts {
                    // Skip thought/reasoning parts
                    if part.thought == Some(true) {
                        continue;
                    }
                    if let Some(text) = &part.text {
                        text_parts.push(text.clone());
                    }
                    if let Some(fc) = &part.function_call {
                        let call_id = fc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                        tool_calls.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": fc.args.to_string()
                            }
                        }));
                    }
                }
            }
        }
    }

    // Override finish_reason when function calls are present
    if !tool_calls.is_empty() {
        finish_reason = "tool_calls".to_string();
    }

    let content_value = if text_parts.is_empty() {
        serde_json::Value::Null
    } else {
        json!(text_parts.join(""))
    };

    let tool_calls_value = if tool_calls.is_empty() {
        serde_json::Value::Null
    } else {
        json!(tool_calls)
    };

    let (prompt_tokens, completion_tokens, total_tokens) =
        if let Some(meta) = &response.usage_metadata {
            (
                meta.prompt_token_count.unwrap_or(0) as i64,
                meta.candidates_token_count.unwrap_or(0) as i64,
                meta.total_token_count.unwrap_or(0) as i64,
            )
        } else {
            (0i64, 0i64, 0i64)
        };

    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content_value,
                "tool_calls": tool_calls_value
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    })
}

// ─── Public: Gemini request → OpenAI Chat request ────────────────────────────

/// Gemini generateContent request → OpenAI ChatCompletion request.
///
/// Uses a serde_json round-trip so we don't have to construct all async_openai
/// builder types by hand.
pub(crate) fn convert_gemini_to_openai_chat_request(
    model: String,
    is_stream: bool,
    req: GeminiGenerateContentRequest,
) -> CreateChatCompletionRequest {
    let mut messages_json: Vec<serde_json::Value> = Vec::new();
    let mut pending_tool_call_ids: HashMap<String, VecDeque<String>> = HashMap::new();

    // System instruction → prepended system message
    if let Some(sys) = req.system_instruction {
        let text: String = sys
            .parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            messages_json.push(json!({
                "role": "system",
                "content": text
            }));
        }
    }

    for content in req.contents {
        let role = content.role.as_deref().unwrap_or("user");

        match role {
            "model" => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls_json: Vec<serde_json::Value> = Vec::new();

                for part in &content.parts {
                    if let Some(text) = &part.text {
                        if part.thought != Some(true) {
                            text_parts.push(text.clone());
                        }
                    }
                    if let Some(fc) = &part.function_call {
                        let call_id = fc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                        pending_tool_call_ids
                            .entry(fc.name.clone())
                            .or_default()
                            .push_back(call_id.clone());
                        tool_calls_json.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": fc.args.to_string()
                            }
                        }));
                    }
                }

                let mut msg = json!({ "role": "assistant" });
                let combined_text = text_parts.join("");
                if !combined_text.is_empty() {
                    msg["content"] = json!(combined_text);
                }
                if !tool_calls_json.is_empty() {
                    msg["tool_calls"] = json!(tool_calls_json);
                }
                messages_json.push(msg);
            }
            _ => {
                // user role — may contain text parts and/or function_response parts
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_msgs: Vec<serde_json::Value> = Vec::new();

                for part in &content.parts {
                    if let Some(text) = &part.text {
                        text_parts.push(text.clone());
                    }
                    if let Some(fr) = &part.function_response {
                        // Serialise the response object back to a JSON string so OpenAI sees it
                        // as the tool's text output.
                        let content_str = fr.response.to_string();
                        let tool_call_id = pending_tool_call_ids
                            .get_mut(&fr.name)
                            .and_then(|ids| ids.pop_front())
                            .unwrap_or_else(|| fr.name.clone());
                        tool_msgs.push(json!({
                            "role": "tool",
                            "content": content_str,
                            "tool_call_id": tool_call_id
                        }));
                    }
                }

                if !text_parts.is_empty() {
                    messages_json.push(json!({
                        "role": "user",
                        "content": text_parts.join("")
                    }));
                }
                messages_json.extend(tool_msgs);
            }
        }
    }

    // Tools
    let tools_json: Option<serde_json::Value> = req.tools.and_then(|tools| {
        let tool_list: Vec<serde_json::Value> = tools
            .into_iter()
            .flat_map(|t| t.function_declarations.unwrap_or_default())
            .map(|fd| {
                json!({
                    "type": "function",
                    "function": {
                        "name": fd.name,
                        "description": fd.description,
                        "parameters": fd.parameters
                    }
                })
            })
            .collect();
        if tool_list.is_empty() {
            None
        } else {
            Some(json!(tool_list))
        }
    });

    let mut request_json = json!({
        "model": model,
        "messages": messages_json,
        "stream": is_stream,
    });

    if is_stream {
        request_json["stream_options"] = json!({
            "include_usage": true
        });
    }

    if let Some(gen_config) = req.generation_config {
        if let Some(max_tokens) = gen_config.max_output_tokens {
            request_json["max_completion_tokens"] = json!(max_tokens);
        }
        if let Some(temp) = gen_config.temperature {
            request_json["temperature"] = json!(temp);
        }
        if let Some(top_p) = gen_config.top_p {
            request_json["top_p"] = json!(top_p);
        }
        if let Some(stops) = gen_config.stop_sequences {
            if !stops.is_empty() {
                request_json["stop"] = json!(stops);
            }
        }
    }

    if let Some(tools) = tools_json {
        request_json["tools"] = tools;
    }

    serde_json::from_value(request_json).unwrap_or_else(|_| CreateChatCompletionRequest {
        model: "unknown".to_string(),
        messages: Vec::new(),
        stream_options: if is_stream {
            Some(ChatCompletionStreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            })
        } else {
            None
        },
        ..Default::default()
    })
}

// ─── Public: OpenAI Chat response → Gemini response ──────────────────────────

/// OpenAI ChatCompletion response JSON → Gemini generateContent response.
pub(crate) fn convert_openai_chat_to_gemini_response(
    response: &serde_json::Value,
    model: &str,
) -> GeminiGenerateContentResponse {
    let mut parts: Vec<GeminiPart> = Vec::new();
    let mut finish_reason = "STOP".to_string();

    let message = response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|choice| {
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                finish_reason = match fr {
                    "stop" => "STOP",
                    "length" => "MAX_TOKENS",
                    _ => "STOP",
                }
                .to_string();
            }
            choice.get("message")
        });

    if let Some(msg) = message {
        // Text content
        if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                parts.push(make_text_part(text.to_string()));
            }
        }

        // Tool calls → function_call parts
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let args_str = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");
                let args: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));

                parts.push(GeminiPart {
                    function_call: Some(GeminiFunctionCall { name, args, id }),
                    ..empty_gemini_part()
                });
            }
        }
    }

    let usage_metadata = response.get("usage").map(|usage| GeminiUsageMetadata {
        prompt_token_count: usage
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        cached_content_token_count: None,
        candidates_token_count: usage
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        total_token_count: usage
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    });

    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts,
            }),
            finish_reason: Some(finish_reason),
            finish_message: None,
            index: Some(0),
        }]),
        usage_metadata,
        model_version: Some(model.to_string()),
        response_id: response
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_part(text: &str) -> GeminiPart {
        GeminiPart {
            text: Some(text.to_string()),
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            cache_control: None,
        }
    }

    #[test]
    fn gemini_stream_request_enables_openai_usage_streaming() {
        let req = GeminiGenerateContentRequest {
            contents: vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![text_part("hello")],
            }],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };

        let converted =
            convert_gemini_to_openai_chat_request("gpt-5.4-nano".to_string(), true, req);

        assert_eq!(converted.stream, Some(true));
        assert_eq!(
            converted
                .stream_options
                .as_ref()
                .and_then(|opts| opts.include_usage),
            Some(true)
        );
    }

    #[test]
    fn gemini_function_response_uses_previous_function_call_id() {
        let req = GeminiGenerateContentRequest {
            contents: vec![
                GeminiContent {
                    role: Some("user".to_string()),
                    parts: vec![text_part("show files")],
                },
                GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart {
                        function_call: Some(GeminiFunctionCall {
                            name: "list_directory".to_string(),
                            args: json!({ "dir_path": "/tmp/demo" }),
                            id: Some("call_123".to_string()),
                        }),
                        ..empty_gemini_part()
                    }],
                },
                GeminiContent {
                    role: Some("user".to_string()),
                    parts: vec![GeminiPart {
                        function_response: Some(GeminiFunctionResponse {
                            name: "list_directory".to_string(),
                            response: json!({ "output": "file_a\nfile_b" }),
                        }),
                        ..empty_gemini_part()
                    }],
                },
            ],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };

        let converted =
            convert_gemini_to_openai_chat_request("gpt-5.4-nano".to_string(), true, req);

        let messages = serde_json::to_value(&converted)
            .unwrap()
            .get("messages")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap();

        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_123");
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_123");
    }

    #[test]
    fn gemini_multiple_same_function_responses_consume_call_ids_in_order() {
        let req = GeminiGenerateContentRequest {
            contents: vec![
                GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![
                        GeminiPart {
                            function_call: Some(GeminiFunctionCall {
                                name: "list_directory".to_string(),
                                args: json!({ "dir_path": "/tmp/a" }),
                                id: Some("call_a".to_string()),
                            }),
                            ..empty_gemini_part()
                        },
                        GeminiPart {
                            function_call: Some(GeminiFunctionCall {
                                name: "list_directory".to_string(),
                                args: json!({ "dir_path": "/tmp/b" }),
                                id: Some("call_b".to_string()),
                            }),
                            ..empty_gemini_part()
                        },
                    ],
                },
                GeminiContent {
                    role: Some("user".to_string()),
                    parts: vec![
                        GeminiPart {
                            function_response: Some(GeminiFunctionResponse {
                                name: "list_directory".to_string(),
                                response: json!({ "output": "a" }),
                            }),
                            ..empty_gemini_part()
                        },
                        GeminiPart {
                            function_response: Some(GeminiFunctionResponse {
                                name: "list_directory".to_string(),
                                response: json!({ "output": "b" }),
                            }),
                            ..empty_gemini_part()
                        },
                    ],
                },
            ],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };

        let converted =
            convert_gemini_to_openai_chat_request("gpt-5.4-nano".to_string(), false, req);

        let messages = serde_json::to_value(&converted)
            .unwrap()
            .get("messages")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap();

        assert_eq!(messages[1]["tool_call_id"], "call_a");
        assert_eq!(messages[2]["tool_call_id"], "call_b");
    }
}
