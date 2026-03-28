use crate::types::{
    AnthropicRequest, CacheControl, ResponseInputContentPart, ResponseInputItem,
    ResponseMessageInputContent, ResponseToolChoice, ResponsesRequest,
};
use serde_json::json;
use tracing::debug;

pub(crate) fn convert_responses_to_anthropic(request: &ResponsesRequest) -> AnthropicRequest {
    let mut messages = Vec::new();
    let mut tools = request.tools.clone();
    let mut pending_assistant_blocks: Vec<crate::types::ContentBlock> = Vec::new();

    let tool_choice = match &request.tool_choice {
        Some(ResponseToolChoice::Mode(mode)) => match mode.as_str() {
            "auto" => Some(crate::types::ToolChoice::Auto),
            "required" => Some(crate::types::ToolChoice::Any),
            "none" => {
                tools = None;
                None
            }
            _ => None,
        },
        Some(ResponseToolChoice::Named { choice_type, name }) if choice_type == "function" => {
            Some(crate::types::ToolChoice::Tool { name: name.clone() })
        }
        Some(ResponseToolChoice::AllowedTools {
            choice_type,
            mode,
            tools: allowed_tools,
        }) if choice_type == "allowed_tools" => {
            if let Some(existing_tools) = &mut tools {
                existing_tools.retain(|tool| {
                    let tool_name = tool.name.as_deref();
                    allowed_tools.iter().any(|allowed| {
                        allowed.tool_type == "function" && tool_name == Some(allowed.name.as_str())
                    })
                });
            }
            match mode.as_str() {
                "required" => Some(crate::types::ToolChoice::Any),
                "auto" => Some(crate::types::ToolChoice::Auto),
                "none" => {
                    tools = None;
                    None
                }
                _ => None,
            }
        }
        _ => None,
    };

    let flush_pending_assistant_blocks =
        |messages: &mut Vec<crate::types::Message>,
         pending_assistant_blocks: &mut Vec<crate::types::ContentBlock>| {
            if !pending_assistant_blocks.is_empty() {
                messages.push(crate::types::Message {
                    role: "assistant".to_string(),
                    content: crate::types::MessageContent::Blocks(std::mem::take(
                        pending_assistant_blocks,
                    )),
                });
            }
        };

    if let Some(ref input) = request.input {
        if input.is_string() {
            flush_pending_assistant_blocks(&mut messages, &mut pending_assistant_blocks);
            if let Some(text) = input.as_str() {
                messages.push(crate::types::Message {
                    role: "user".to_string(),
                    content: crate::types::MessageContent::String(text.to_string()),
                });
            }
        } else if let Some(arr) = input.as_array() {
            for value in arr {
                if value.get("role").is_some() {
                    let role = value.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let mapped_role = if role == "developer" { "user" } else { role };
                    let content_value = value.get("content");
                    if let Some(content_val) = content_value {
                        let message_content =
                            convert_response_content_to_message_content(content_val);
                        if !is_message_content_empty(&message_content) {
                            if mapped_role == "assistant" {
                                match message_content {
                                    crate::types::MessageContent::Blocks(blocks) => {
                                        pending_assistant_blocks.extend(blocks);
                                    }
                                    crate::types::MessageContent::String(text) => {
                                        pending_assistant_blocks.push(
                                            crate::types::ContentBlock::Text {
                                                text,
                                                cache_control: None,
                                            },
                                        );
                                    }
                                }
                            } else {
                                flush_pending_assistant_blocks(
                                    &mut messages,
                                    &mut pending_assistant_blocks,
                                );
                                messages.push(crate::types::Message {
                                    role: mapped_role.to_string(),
                                    content: message_content,
                                });
                            }
                        }
                    }
                    continue;
                }

                if let Ok(item) = serde_json::from_value::<ResponseInputItem>(value.clone()) {
                    match item {
                        ResponseInputItem::Message { role, content } => {
                            let mapped_role = if role == "developer" {
                                "user"
                            } else {
                                role.as_str()
                            };
                            let message_content = convert_response_message_input_content(content);
                            if !is_message_content_empty(&message_content) {
                                if mapped_role == "assistant" {
                                    match message_content {
                                        crate::types::MessageContent::Blocks(blocks) => {
                                            pending_assistant_blocks.extend(blocks);
                                        }
                                        crate::types::MessageContent::String(text) => {
                                            pending_assistant_blocks.push(
                                                crate::types::ContentBlock::Text {
                                                    text,
                                                    cache_control: None,
                                                },
                                            );
                                        }
                                    }
                                } else {
                                    flush_pending_assistant_blocks(
                                        &mut messages,
                                        &mut pending_assistant_blocks,
                                    );
                                    messages.push(crate::types::Message {
                                        role: mapped_role.to_string(),
                                        content: message_content,
                                    });
                                }
                            }
                        }
                        ResponseInputItem::FunctionCall {
                            id,
                            call_id,
                            name,
                            arguments,
                            thought_signature,
                        } => {
                            debug!(
                                id = id.as_deref().unwrap_or(call_id.as_str()),
                                call_id = %call_id,
                                name = %name,
                                has_thought_signature = thought_signature.is_some(),
                                "Parsing responses function_call input item"
                            );
                            let tool_input =
                                serde_json::from_str(&arguments).unwrap_or_else(|_| json!({}));
                            let tool_use_id = id.unwrap_or_else(|| call_id.clone());
                            pending_assistant_blocks.push(crate::types::ContentBlock::ToolUse {
                                id: tool_use_id,
                                name,
                                input: tool_input,
                                thought_signature,
                                cache_control: None,
                            });
                        }
                        ResponseInputItem::FunctionCallOutput { call_id, output } => {
                            flush_pending_assistant_blocks(
                                &mut messages,
                                &mut pending_assistant_blocks,
                            );
                            messages.push(crate::types::Message {
                                role: "user".to_string(),
                                content: crate::types::MessageContent::Blocks(vec![
                                    crate::types::ContentBlock::ToolResult {
                                        tool_use_id: call_id,
                                        content: Some(convert_function_output_to_tool_result(
                                            output,
                                        )),
                                        is_error: None,
                                        cache_control: None,
                                    },
                                ]),
                            });
                        }
                        ResponseInputItem::Reasoning { summary, .. } => {
                            let thinking = summary
                                .unwrap_or_default()
                                .into_iter()
                                .map(|part| match part {
                                    crate::types::ResponseReasoningSummary::SummaryText {
                                        text,
                                    } => text,
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            if !thinking.is_empty() {
                                pending_assistant_blocks.push(
                                    crate::types::ContentBlock::Thinking {
                                        thinking,
                                        signature: None,
                                    },
                                );
                            }
                        }
                    }
                    continue;
                }

                if let Some(content_block) = convert_json_to_content_block(value) {
                    flush_pending_assistant_blocks(&mut messages, &mut pending_assistant_blocks);
                    messages.push(crate::types::Message {
                        role: "user".to_string(),
                        content: crate::types::MessageContent::Blocks(vec![content_block]),
                    });
                }
            }
        }
    }

    flush_pending_assistant_blocks(&mut messages, &mut pending_assistant_blocks);

    let system = request
        .instructions
        .as_ref()
        .map(|i| crate::types::SystemPrompt::String(i.clone()));

    let tools = tools.as_ref().map(|tools_vec| {
        tools_vec
            .iter()
            .filter_map(|t| {
                let name = t.name.as_ref()?;
                Some(crate::types::Tool {
                    name: name.clone(),
                    description: t.description.clone(),
                    input_schema: t.parameters.clone().unwrap_or(serde_json::json!({})),
                })
            })
            .collect()
    });

    AnthropicRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_output_tokens.unwrap_or(1024),
        metadata: request.metadata.clone(),
        stop_sequences: None,
        stream: request.stream,
        system,
        temperature: request.temperature,
        thinking: None,
        tool_choice,
        tools,
        top_k: None,
        top_p: None,
        cache_control: None,
    }
}

fn convert_response_message_input_content(
    content: ResponseMessageInputContent,
) -> crate::types::MessageContent {
    match content {
        ResponseMessageInputContent::String(text) => crate::types::MessageContent::String(text),
        ResponseMessageInputContent::Blocks(parts) => {
            let mut blocks = Vec::new();
            for part in parts {
                match part {
                    ResponseInputContentPart::InputText {
                        text,
                        cache_control,
                    }
                    | ResponseInputContentPart::OutputText {
                        text,
                        cache_control,
                    }
                    | ResponseInputContentPart::Text {
                        text,
                        cache_control,
                    } => {
                        blocks.push(crate::types::ContentBlock::Text {
                            text,
                            cache_control,
                        });
                    }
                    ResponseInputContentPart::InputImage {
                        image_url,
                        cache_control,
                        ..
                    } => {
                        if let Some(url) = image_url {
                            blocks.push(crate::types::ContentBlock::Image {
                                source: parse_response_input_image_source(&url),
                                cache_control,
                            });
                        }
                    }
                }
            }
            crate::types::MessageContent::Blocks(blocks)
        }
    }
}

fn convert_response_content_to_message_content(
    content_val: &serde_json::Value,
) -> crate::types::MessageContent {
    if let Some(text) = content_val.as_str() {
        return crate::types::MessageContent::String(text.to_string());
    }
    if let Some(arr) = content_val.as_array() {
        let blocks: Vec<crate::types::ContentBlock> = arr
            .iter()
            .filter_map(convert_json_to_content_block)
            .collect();
        if !blocks.is_empty() {
            return crate::types::MessageContent::Blocks(blocks);
        }
    }
    crate::types::MessageContent::String(String::new())
}

fn parse_response_input_image_source(url: &str) -> crate::types::ImageSource {
    if let Some(data) = url.strip_prefix("data:") {
        if let Some((header, payload)) = data.split_once(',') {
            if let Some(media_type) = header.strip_suffix(";base64") {
                return crate::types::ImageSource::Base64 {
                    media_type: media_type.to_string(),
                    data: payload.to_string(),
                };
            }
        }
    }

    crate::types::ImageSource::Url {
        url: url.to_string(),
    }
}

fn convert_function_output_to_tool_result(
    output: serde_json::Value,
) -> crate::types::ToolResultMessageContent {
    match output {
        serde_json::Value::String(text) => crate::types::ToolResultMessageContent::String(text),
        serde_json::Value::Array(items) => {
            let blocks = items
                .iter()
                .filter_map(
                    |item| match item.get("type").and_then(|value| value.as_str()) {
                        Some("text") => Some(crate::types::ToolResultContentBlock::Text {
                            text: item
                                .get("text")
                                .and_then(|value| value.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            cache_control: None,
                        }),
                        Some("image") => item.get("source").and_then(|source| {
                            let url = source.get("url").and_then(|value| value.as_str())?;
                            Some(crate::types::ToolResultContentBlock::Image {
                                source: crate::types::ImageSource::Url {
                                    url: url.to_string(),
                                },
                                cache_control: None,
                            })
                        }),
                        _ => None,
                    },
                )
                .collect::<Vec<_>>();
            if blocks.is_empty() {
                crate::types::ToolResultMessageContent::String(
                    serde_json::Value::Array(items).to_string(),
                )
            } else {
                crate::types::ToolResultMessageContent::Blocks(blocks)
            }
        }
        other => crate::types::ToolResultMessageContent::String(other.to_string()),
    }
}

fn is_message_content_empty(content: &crate::types::MessageContent) -> bool {
    match content {
        crate::types::MessageContent::String(s) => s.is_empty(),
        crate::types::MessageContent::Blocks(blocks) => blocks.is_empty(),
    }
}

fn convert_json_to_content_block(val: &serde_json::Value) -> Option<crate::types::ContentBlock> {
    let block_type = val.get("type")?.as_str()?;
    let cache_control = val
        .get("cache_control")
        .and_then(|value| serde_json::from_value::<CacheControl>(value.clone()).ok());

    match block_type {
        "text" | "input_text" => {
            let text = val.get("text").and_then(|t| t.as_str()).unwrap_or("");
            Some(crate::types::ContentBlock::Text {
                text: text.to_string(),
                cache_control,
            })
        }
        "image" => {
            let source = if let Some(source_val) = val.get("source") {
                if let Some(url) = source_val.get("url").and_then(|u| u.as_str()) {
                    parse_response_input_image_source(url)
                } else if let (Some(media_type), Some(data)) = (
                    source_val.get("media_type").and_then(|m| m.as_str()),
                    source_val.get("data").and_then(|d| d.as_str()),
                ) {
                    crate::types::ImageSource::Base64 {
                        media_type: media_type.to_string(),
                        data: data.to_string(),
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            };
            Some(crate::types::ContentBlock::Image {
                source,
                cache_control,
            })
        }
        "input_image" => {
            let url = val.get("image_url").and_then(|value| value.as_str())?;
            Some(crate::types::ContentBlock::Image {
                source: parse_response_input_image_source(url),
                cache_control,
            })
        }
        "thinking" => {
            let thinking = val
                .get("thinking")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            Some(crate::types::ContentBlock::Thinking {
                thinking,
                signature: None,
            })
        }
        "refusal" => {
            let content = val
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            Some(crate::types::ContentBlock::Text {
                text: content,
                cache_control,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::convert_responses_to_anthropic;
    use crate::types::{ImageSource, MessageContent, ResponsesRequest};

    #[test]
    fn preserves_cache_control_when_converting_responses_to_anthropic() {
        let request = ResponsesRequest {
            model: "test-model".to_string(),
            input: Some(serde_json::json!([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "cache me",
                            "cache_control": {
                                "type": "ephemeral"
                            }
                        }
                    ]
                }
            ])),
            previous_response_id: None,
            instructions: None,
            stream: Some(false),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_output_tokens: None,
            metadata: None,
        };

        let anthropic = convert_responses_to_anthropic(&request);
        let cache_control = anthropic
            .messages
            .first()
            .and_then(|message| match &message.content {
                MessageContent::Blocks(blocks) => blocks.first(),
                MessageContent::String(_) => None,
            })
            .and_then(|block| match block {
                crate::types::ContentBlock::Text { cache_control, .. } => cache_control.as_ref(),
                _ => None,
            });

        assert_eq!(
            cache_control.map(|value| value.control_type.as_str()),
            Some("ephemeral")
        );
    }

    #[test]
    fn preserves_input_image_blocks_for_role_content_inputs() {
        let request = ResponsesRequest {
            model: "test-model".to_string(),
            input: Some(serde_json::json!([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "look"
                        },
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.png"
                        }
                    ]
                }
            ])),
            previous_response_id: None,
            instructions: None,
            stream: Some(false),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_output_tokens: None,
            metadata: None,
        };

        let anthropic = convert_responses_to_anthropic(&request);
        let MessageContent::Blocks(blocks) = &anthropic.messages[0].content else {
            panic!("expected block content");
        };

        let crate::types::ContentBlock::Image { source, .. } = &blocks[1] else {
            panic!("expected image block");
        };

        match source {
            ImageSource::Url { url } => assert_eq!(url, "https://example.com/image.png"),
            other => panic!("expected url image source, got {other:?}"),
        }
    }

    #[test]
    fn parses_data_urls_in_response_input_images_as_base64() {
        let request = ResponsesRequest {
            model: "test-model".to_string(),
            input: Some(serde_json::json!([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,Zm9vYmFy"
                        }
                    ]
                }
            ])),
            previous_response_id: None,
            instructions: None,
            stream: Some(false),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_output_tokens: None,
            metadata: None,
        };

        let anthropic = convert_responses_to_anthropic(&request);
        let MessageContent::Blocks(blocks) = &anthropic.messages[0].content else {
            panic!("expected block content");
        };

        let crate::types::ContentBlock::Image { source, .. } = &blocks[0] else {
            panic!("expected image block");
        };

        match source {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "Zm9vYmFy");
            }
            other => panic!("expected base64 image source, got {other:?}"),
        }
    }

    #[test]
    fn preserves_thought_signature_on_function_call_inputs() {
        let request = ResponsesRequest {
            model: "test-model".to_string(),
            input: Some(serde_json::json!([
                {
                    "type": "function_call",
                    "id": "call_1",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"ls -la\"}",
                    "thought_signature": "sig-123"
                }
            ])),
            previous_response_id: None,
            instructions: None,
            stream: Some(false),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_output_tokens: None,
            metadata: None,
        };

        let anthropic = convert_responses_to_anthropic(&request);
        let MessageContent::Blocks(blocks) = &anthropic.messages[0].content else {
            panic!("expected block content");
        };

        let crate::types::ContentBlock::ToolUse {
            thought_signature, ..
        } = &blocks[0]
        else {
            panic!("expected tool use block");
        };

        assert_eq!(thought_signature.as_deref(), Some("sig-123"));
    }
}
