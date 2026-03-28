use crate::types::{
    GeminiContent, GeminiFileData, GeminiFunctionCall, GeminiFunctionDeclaration,
    GeminiFunctionResponse, GeminiGenerateContentRequest, GeminiGenerateContentResponse,
    GeminiGenerationConfig, GeminiInlineData, GeminiPart, GeminiTool, ResponseInputContentPart,
    ResponseInputItem, ResponseMessageInputContent, ResponseOutputContent, ResponseOutputItem,
    ResponseReasoningSummary, ResponseUsage, ResponsesRequest, ResponsesResponse,
};

pub(crate) fn convert_responses_to_gemini_request(
    request: &ResponsesRequest,
) -> GeminiGenerateContentRequest {
    let system_instruction = request
        .instructions
        .as_ref()
        .map(|instructions| GeminiContent {
            role: None,
            parts: vec![GeminiPart {
                text: Some(instructions.clone()),
                thought: None,
                thought_signature: None,
                inline_data: None,
                file_data: None,
                function_call: None,
                function_response: None,
                cache_control: None,
            }],
        });

    let contents = convert_input_to_gemini_contents(&request.input);

    let tools = request.tools.as_ref().and_then(|tools_vec| {
        if tools_vec.is_empty() {
            return None;
        }
        let declarations: Vec<GeminiFunctionDeclaration> = tools_vec
            .iter()
            .filter_map(|t| {
                let name = t.name.as_ref()?.clone();
                Some(GeminiFunctionDeclaration {
                    name,
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                })
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

    let generation_config = if request.max_output_tokens.is_some() || request.temperature.is_some()
    {
        Some(GeminiGenerationConfig {
            max_output_tokens: request.max_output_tokens,
            temperature: request.temperature,
            top_p: None,
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

fn convert_input_to_gemini_contents(input: &Option<serde_json::Value>) -> Vec<GeminiContent> {
    let input_array = match input {
        Some(v) if v.is_string() => {
            return vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart {
                    text: v.as_str().map(|s| s.to_string()),
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    cache_control: None,
                }],
            }];
        }
        Some(v) if v.is_array() => v.as_array().unwrap().clone(),
        _ => return vec![],
    };

    let mut contents: Vec<GeminiContent> = Vec::new();

    for value in &input_array {
        let item = match serde_json::from_value::<ResponseInputItem>(value.clone()) {
            Ok(item) => item,
            Err(_) => {
                // Fall back to handling plain message objects with role/content
                if let Some(role) = value.get("role").and_then(|r| r.as_str()) {
                    let gemini_role = map_role_to_gemini(role);
                    let parts = if let Some(content) = value.get("content") {
                        convert_content_value_to_parts(content)
                    } else {
                        vec![]
                    };
                    if !parts.is_empty() {
                        push_or_merge_content(&mut contents, gemini_role, parts);
                    }
                }
                continue;
            }
        };

        match item {
            ResponseInputItem::Message { role, content } => {
                let gemini_role = map_role_to_gemini(&role);
                let parts = convert_message_content_to_parts(content);
                if !parts.is_empty() {
                    push_or_merge_content(&mut contents, gemini_role, parts);
                }
            }
            ResponseInputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
                thought_signature,
            } => {
                let args: serde_json::Value = serde_json::from_str(&arguments)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
                let fc_id = id.or(Some(call_id));
                let part = GeminiPart {
                    text: None,
                    thought: None,
                    thought_signature,
                    inline_data: None,
                    file_data: None,
                    function_call: Some(GeminiFunctionCall {
                        name,
                        args,
                        id: fc_id,
                    }),
                    function_response: None,
                    cache_control: None,
                };
                push_or_merge_content(&mut contents, "model".to_string(), vec![part]);
            }
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                let response_body = if output.is_object() {
                    output
                } else {
                    serde_json::json!({ "content": output })
                };
                let part = GeminiPart {
                    text: None,
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: Some(GeminiFunctionResponse {
                        name: call_id.clone(),
                        response: response_body,
                    }),
                    cache_control: None,
                };
                push_or_merge_content(&mut contents, "user".to_string(), vec![part]);
            }
            ResponseInputItem::Reasoning { summary, .. } => {
                let combined_text = summary
                    .unwrap_or_default()
                    .into_iter()
                    .map(|part| match part {
                        ResponseReasoningSummary::SummaryText { text } => text,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if !combined_text.is_empty() {
                    let part = GeminiPart {
                        text: Some(combined_text),
                        thought: Some(true),
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: None,
                        function_response: None,
                        cache_control: None,
                    };
                    push_or_merge_content(&mut contents, "model".to_string(), vec![part]);
                }
            }
        }
    }

    contents
}

fn map_role_to_gemini(role: &str) -> String {
    match role {
        "assistant" => "model".to_string(),
        "developer" => "user".to_string(),
        other => other.to_string(),
    }
}

/// Append parts to the last GeminiContent if it shares the same role; otherwise push a new one.
fn push_or_merge_content(contents: &mut Vec<GeminiContent>, role: String, parts: Vec<GeminiPart>) {
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

fn convert_message_content_to_parts(content: ResponseMessageInputContent) -> Vec<GeminiPart> {
    match content {
        ResponseMessageInputContent::String(text) => vec![GeminiPart {
            text: Some(text),
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            cache_control: None,
        }],
        ResponseMessageInputContent::Blocks(parts) => parts
            .into_iter()
            .filter_map(convert_input_content_part_to_gemini_part)
            .collect(),
    }
}

fn convert_content_value_to_parts(value: &serde_json::Value) -> Vec<GeminiPart> {
    if let Some(text) = value.as_str() {
        return vec![GeminiPart {
            text: Some(text.to_string()),
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            cache_control: None,
        }];
    }

    if let Some(arr) = value.as_array() {
        return arr
            .iter()
            .filter_map(|part_val| {
                if let Ok(part) =
                    serde_json::from_value::<ResponseInputContentPart>(part_val.clone())
                {
                    convert_input_content_part_to_gemini_part(part)
                } else if let Some(text) = part_val.get("text").and_then(|t| t.as_str()) {
                    Some(GeminiPart {
                        text: Some(text.to_string()),
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: None,
                        function_response: None,
                        cache_control: None,
                    })
                } else {
                    None
                }
            })
            .collect();
    }

    vec![]
}

fn convert_input_content_part_to_gemini_part(part: ResponseInputContentPart) -> Option<GeminiPart> {
    match part {
        ResponseInputContentPart::InputText { text, .. }
        | ResponseInputContentPart::OutputText { text, .. }
        | ResponseInputContentPart::Text { text, .. } => Some(GeminiPart {
            text: Some(text),
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            cache_control: None,
        }),
        ResponseInputContentPart::InputImage { image_url, .. } => {
            let url = image_url?;
            // Check if it's a data URL (base64 encoded)
            if let Some(rest) = url.strip_prefix("data:") {
                if let Some((mime_and_enc, data)) = rest.split_once(',') {
                    let mime_type = mime_and_enc
                        .split(';')
                        .next()
                        .unwrap_or("image/jpeg")
                        .to_string();
                    Some(GeminiPart {
                        text: None,
                        thought: None,
                        thought_signature: None,
                        inline_data: Some(GeminiInlineData {
                            mime_type,
                            data: data.to_string(),
                        }),
                        file_data: None,
                        function_call: None,
                        function_response: None,
                        cache_control: None,
                    })
                } else {
                    None
                }
            } else {
                // Treat as a file URI
                Some(GeminiPart {
                    text: None,
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: Some(GeminiFileData {
                        mime_type: "image/jpeg".to_string(),
                        file_uri: url,
                    }),
                    function_call: None,
                    function_response: None,
                    cache_control: None,
                })
            }
        }
    }
}

pub(crate) fn convert_gemini_response_to_responses(
    model: &str,
    response: GeminiGenerateContentResponse,
) -> ResponsesResponse {
    let mut message_content: Vec<ResponseOutputContent> = Vec::new();
    let mut reasoning_summaries: Vec<ResponseReasoningSummary> = Vec::new();
    let mut function_calls: Vec<ResponseOutputItem> = Vec::new();

    if let Some(candidates) = response.candidates {
        if let Some(candidate) = candidates.into_iter().next() {
            if let Some(content) = candidate.content {
                for part in content.parts {
                    if part.thought == Some(true) {
                        if let Some(text) = part.text {
                            reasoning_summaries
                                .push(ResponseReasoningSummary::SummaryText { text });
                        }
                    } else if let Some(fc) = part.function_call {
                        let fc_id = fc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                        function_calls.push(ResponseOutputItem::FunctionCall {
                            id: fc_id.clone(),
                            status: Some("completed".to_string()),
                            call_id: fc_id,
                            name: fc.name,
                            arguments: fc.args.to_string(),
                            thought_signature: part.thought_signature,
                        });
                    } else if let Some(text) = part.text {
                        message_content.push(ResponseOutputContent::OutputText {
                            text,
                            annotations: Some(vec![]),
                        });
                    }
                }
            }
        }
    }

    let mut output: Vec<ResponseOutputItem> = Vec::new();

    if !reasoning_summaries.is_empty() {
        output.push(ResponseOutputItem::Reasoning {
            id: format!("rs_{}", uuid::Uuid::new_v4()),
            summary: Some(reasoning_summaries),
        });
    }

    if !message_content.is_empty() {
        output.push(ResponseOutputItem::Message {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            status: Some("completed".to_string()),
            role: "assistant".to_string(),
            content: message_content,
        });
    }

    output.extend(function_calls);

    let usage = response.usage_metadata.map(|meta| ResponseUsage {
        input_tokens: Some(meta.prompt_token_count.unwrap_or(0)),
        output_tokens: Some(meta.candidates_token_count.unwrap_or(0)),
        total_tokens: Some(meta.total_token_count.unwrap_or(0)),
    });

    ResponsesResponse {
        id: format!("resp_{}", uuid::Uuid::new_v4()),
        object: "response".to_string(),
        model: model.to_string(),
        created_at: chrono::Utc::now().timestamp(),
        output: Some(output),
        usage,
    }
}
