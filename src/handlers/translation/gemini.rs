use crate::types::{
    AnthropicRequest, AnthropicResponse, ContentBlock, ErrorDetails, GeminiCandidate,
    GeminiContent, GeminiFileData, GeminiFunctionCall, GeminiFunctionDeclaration,
    GeminiFunctionResponse, GeminiGenerateContentRequest, GeminiGenerateContentResponse,
    GeminiInlineData, GeminiPart, GeminiTool, GeminiUsageMetadata, ImageSource, Message,
    MessageContent, MessageDelta, MessageDeltaUsage, StreamEvent, SystemPrompt, Tool,
    ToolResultMessageContent, Usage,
};
use serde_json::Value;
use std::collections::HashMap;
use tracing::debug;

#[derive(Debug, Default)]
pub(crate) struct GeminiStreamState {
    pub message_started: bool,
    pub next_index: u32,
    pub emitted_message_stop: bool,
    pub pending_thought_signature: Option<String>,
}

pub(crate) fn convert_gemini_to_anthropic(
    model: String,
    is_stream: bool,
    request: GeminiGenerateContentRequest,
) -> AnthropicRequest {
    let GeminiGenerateContentRequest {
        contents,
        system_instruction,
        generation_config,
        tools,
    } = request;

    AnthropicRequest {
        model,
        messages: contents
            .into_iter()
            .map(convert_content_to_message)
            .collect(),
        max_tokens: generation_config
            .as_ref()
            .and_then(|config| config.max_output_tokens)
            .unwrap_or(1024),
        metadata: None,
        stop_sequences: generation_config
            .as_ref()
            .and_then(|config| config.stop_sequences.clone()),
        stream: Some(is_stream),
        system: system_instruction.and_then(convert_system_instruction),
        temperature: generation_config
            .as_ref()
            .and_then(|config| config.temperature),
        thinking: None,
        tool_choice: None,
        tools: convert_tools(tools),
        top_k: None,
        top_p: generation_config.and_then(|config| config.top_p),
        cache_control: None,
    }
}

pub(crate) fn convert_anthropic_to_gemini_request(
    request: AnthropicRequest,
) -> GeminiGenerateContentRequest {
    let mut tool_names = HashMap::new();

    GeminiGenerateContentRequest {
        contents: request
            .messages
            .into_iter()
            .map(|message| convert_message_to_gemini_content(message, &mut tool_names))
            .collect(),
        system_instruction: request.system.and_then(convert_anthropic_system_to_gemini),
        generation_config: Some(crate::types::GeminiGenerationConfig {
            max_output_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop_sequences,
        }),
        tools: request.tools.map(convert_anthropic_tools_to_gemini),
    }
}

pub(crate) fn convert_anthropic_to_gemini(
    response: AnthropicResponse,
) -> GeminiGenerateContentResponse {
    let finish_reason = map_stop_reason(response.stop_reason.as_deref());
    let mut tool_names = HashMap::new();
    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: response
                    .content
                    .into_iter()
                    .filter_map(|block| {
                        convert_content_block_to_gemini_part(block, &mut tool_names)
                    })
                    .collect(),
            }),
            finish_reason: Some(finish_reason),
            finish_message: finish_message_for_stop_reason(response.stop_reason.as_deref()),
            index: Some(0),
        }]),
        usage_metadata: Some(GeminiUsageMetadata {
            prompt_token_count: Some(response.usage.input_tokens),
            cached_content_token_count: response.usage.cache_read_input_tokens,
            candidates_token_count: Some(response.usage.output_tokens),
            total_token_count: Some(response.usage.input_tokens + response.usage.output_tokens),
        }),
        model_version: Some(response.model),
        response_id: Some(response.id),
    }
}

pub(crate) fn convert_gemini_response_to_anthropic(
    fallback_model: &str,
    response: GeminiGenerateContentResponse,
) -> AnthropicResponse {
    let GeminiGenerateContentResponse {
        candidates,
        usage_metadata,
        model_version,
        response_id,
    } = response;

    let candidate = candidates
        .and_then(|mut candidates| candidates.drain(..).next())
        .unwrap_or(GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: Vec::new(),
            }),
            finish_reason: None,
            finish_message: None,
            index: Some(0),
        });

    let stop_reason = candidate.finish_reason.as_deref().map(|reason| {
        map_candidate_finish_reason_to_stop_reason(reason, candidate.content.as_ref())
    });

    let content = candidate
        .content
        .clone()
        .map(convert_content_to_message)
        .map(|message| match message.content {
            MessageContent::Blocks(blocks) => blocks,
            MessageContent::String(text) => vec![ContentBlock::Text {
                text,
                cache_control: None,
            }],
        })
        .unwrap_or_default();

    let usage = usage_metadata.map_or_else(Usage::default, |usage| Usage {
        input_tokens: usage.prompt_token_count.unwrap_or(0),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: usage.cached_content_token_count,
        output_tokens: usage.candidates_token_count.unwrap_or(0),
    });

    AnthropicResponse {
        id: response_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        msg_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model_version.unwrap_or_else(|| fallback_model.to_string()),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

pub(crate) fn convert_gemini_stream_chunk_to_anthropic(
    fallback_model: &str,
    chunk: GeminiGenerateContentResponse,
    state: &mut GeminiStreamState,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if !state.message_started {
        state.message_started = true;
        events.push(StreamEvent::MessageStart {
            message: AnthropicResponse {
                id: chunk
                    .response_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                msg_type: "message".to_string(),
                role: "assistant".to_string(),
                content: Vec::new(),
                model: chunk
                    .model_version
                    .clone()
                    .unwrap_or_else(|| fallback_model.to_string()),
                stop_reason: None,
                stop_sequence: None,
                usage: chunk
                    .usage_metadata
                    .as_ref()
                    .map_or_else(Usage::default, |usage| Usage {
                        input_tokens: usage.prompt_token_count.unwrap_or(0),
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: usage.cached_content_token_count,
                        output_tokens: 0,
                    }),
            },
        });
    }

    if let Some(candidates) = &chunk.candidates {
        for candidate in candidates {
            if let Some(content) = &candidate.content {
                for part in &content.parts {
                    push_gemini_part_stream_events(part, state, &mut events);
                }
            }

            if candidate.finish_reason.is_some() || chunk.usage_metadata.is_some() {
                let usage = chunk.usage_metadata.as_ref().map_or(
                    MessageDeltaUsage {
                        output_tokens: 0,
                        input_tokens: None,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                    },
                    |usage| MessageDeltaUsage {
                        output_tokens: usage.candidates_token_count.unwrap_or(0),
                        input_tokens: usage.prompt_token_count,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: usage.cached_content_token_count,
                    },
                );

                events.push(StreamEvent::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: candidate.finish_reason.as_deref().map(|reason| {
                            map_candidate_finish_reason_to_stop_reason(
                                reason,
                                candidate.content.as_ref(),
                            )
                        }),
                        stop_sequence: None,
                    },
                    usage,
                });
                if candidate.finish_reason.is_some() && !state.emitted_message_stop {
                    state.emitted_message_stop = true;
                    events.push(StreamEvent::MessageStop);
                }
            }
        }
    } else if chunk.usage_metadata.is_some() && !state.emitted_message_stop {
        let usage = chunk
            .usage_metadata
            .as_ref()
            .map(|usage| MessageDeltaUsage {
                output_tokens: usage.candidates_token_count.unwrap_or(0),
                input_tokens: usage.prompt_token_count,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: usage.cached_content_token_count,
            })
            .unwrap();

        events.push(StreamEvent::MessageDelta {
            delta: MessageDelta {
                stop_reason: None,
                stop_sequence: None,
            },
            usage,
        });
        state.emitted_message_stop = true;
        events.push(StreamEvent::MessageStop);
    }

    events
}

pub(crate) fn convert_stream_event_to_gemini(
    event: StreamEvent,
    response_id: &mut Option<String>,
    model_version: &mut Option<String>,
    pending_tool_json: &mut String,
    pending_tool_name: &mut Option<String>,
    pending_tool_id: &mut Option<String>,
) -> Option<GeminiGenerateContentResponse> {
    match event {
        StreamEvent::MessageStart { message } => {
            *response_id = Some(message.id);
            *model_version = Some(message.model);
            None
        }
        StreamEvent::ContentBlockStart { content_block, .. } => match content_block {
            ContentBlock::Text { text, .. } => Some(text_chunk(text)),
            ContentBlock::Thinking {
                thinking,
                signature,
            } => Some(thought_chunk(thinking, signature)),
            ContentBlock::ToolUse { id, name, .. } => {
                *pending_tool_name = Some(name.clone());
                *pending_tool_id = Some(id);
                pending_tool_json.clear();
                None
            }
            _ => None,
        },
        StreamEvent::ContentBlockDelta { delta, .. } => match delta {
            crate::types::ContentBlockDelta::TextDelta { text } => Some(text_chunk(text)),
            crate::types::ContentBlockDelta::ThinkingDelta { thinking } => {
                Some(thought_chunk(thinking, None))
            }
            crate::types::ContentBlockDelta::SignatureDelta { signature } => {
                Some(thought_chunk(String::new(), Some(signature)))
            }
            crate::types::ContentBlockDelta::InputJsonDelta { partial_json } => {
                pending_tool_json.push_str(&partial_json);
                None
            }
        },
        StreamEvent::ContentBlockStop { .. } => {
            if let Some(name) = pending_tool_name.take() {
                let args = if pending_tool_json.is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str::<Value>(pending_tool_json)
                        .unwrap_or_else(|_| Value::String(pending_tool_json.clone()))
                };
                let id = pending_tool_id.take();
                pending_tool_json.clear();
                Some(function_call_chunk(name, args, id))
            } else {
                None
            }
        }
        StreamEvent::MessageDelta { delta, usage } => {
            let stop_reason = delta.stop_reason.as_deref();
            Some(GeminiGenerateContentResponse {
                candidates: Some(vec![GeminiCandidate {
                    content: None,
                    finish_reason: Some(map_stop_reason(stop_reason)),
                    finish_message: finish_message_for_stop_reason(stop_reason),
                    index: Some(0),
                }]),
                usage_metadata: Some(GeminiUsageMetadata {
                    prompt_token_count: Some(usage.input_tokens.unwrap_or(0)),
                    cached_content_token_count: usage.cache_read_input_tokens,
                    candidates_token_count: Some(usage.output_tokens),
                    total_token_count: Some(usage.input_tokens.unwrap_or(0) + usage.output_tokens),
                }),
                model_version: model_version.clone(),
                response_id: response_id.clone(),
            })
        }
        StreamEvent::Error { error } => Some(error_chunk(error)),
        StreamEvent::MessageStop | StreamEvent::Ping => None,
    }
}

fn convert_message_to_gemini_content(
    message: Message,
    tool_names: &mut HashMap<String, String>,
) -> GeminiContent {
    GeminiContent {
        role: Some(
            match message.role.as_str() {
                "assistant" => "model",
                _ => "user",
            }
            .to_string(),
        ),
        parts: match message.content {
            MessageContent::Blocks(blocks) => blocks
                .into_iter()
                .filter_map(|block| convert_content_block_to_gemini_part(block, tool_names))
                .collect(),
            MessageContent::String(text) => vec![GeminiPart {
                text: Some(text),
                thought: None,
                thought_signature: None,
                inline_data: None,
                file_data: None,
                function_call: None,
                function_response: None,
                cache_control: None,
            }],
        },
    }
}

fn convert_content_to_message(content: GeminiContent) -> Message {
    let role = match content.role.as_deref() {
        Some("model") => "assistant",
        _ => "user",
    }
    .to_string();

    let mut pending_thought_signature = None;
    let blocks: Vec<ContentBlock> = content
        .parts
        .into_iter()
        .filter_map(|part| {
            convert_gemini_part_to_content_block(part, &mut pending_thought_signature)
        })
        .collect();

    let content = if blocks.len() == 1 {
        if let ContentBlock::Text {
            text,
            cache_control,
        } = &blocks[0]
        {
            if cache_control.is_none() {
                MessageContent::String(text.clone())
            } else {
                MessageContent::Blocks(blocks)
            }
        } else {
            MessageContent::Blocks(blocks)
        }
    } else {
        MessageContent::Blocks(blocks)
    };

    Message { role, content }
}

fn convert_anthropic_system_to_gemini(system_prompt: SystemPrompt) -> Option<GeminiContent> {
    let mut tool_names = HashMap::new();
    let parts = match system_prompt {
        SystemPrompt::String(text) => vec![GeminiPart {
            text: Some(text),
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            cache_control: None,
        }],
        SystemPrompt::Blocks(blocks) => blocks
            .into_iter()
            .filter_map(|block| convert_content_block_to_gemini_part(block, &mut tool_names))
            .collect(),
    };

    if parts.is_empty() {
        None
    } else {
        Some(GeminiContent { role: None, parts })
    }
}

fn convert_system_instruction(system_instruction: GeminiContent) -> Option<SystemPrompt> {
    let blocks: Vec<ContentBlock> = system_instruction
        .parts
        .into_iter()
        .filter_map(|part| {
            let cache_control = part.cache_control;
            part.text.map(|text| ContentBlock::Text {
                text,
                cache_control,
            })
        })
        .collect();

    if blocks.is_empty() {
        None
    } else {
        Some(SystemPrompt::Blocks(blocks))
    }
}

fn convert_anthropic_tools_to_gemini(tools: Vec<Tool>) -> Vec<GeminiTool> {
    vec![GeminiTool {
        function_declarations: Some(
            tools
                .into_iter()
                .map(|tool| GeminiFunctionDeclaration {
                    name: tool.name,
                    description: tool.description,
                    parameters: Some(tool.input_schema),
                })
                .collect(),
        ),
    }]
}

fn convert_tools(tools: Option<Vec<GeminiTool>>) -> Option<Vec<Tool>> {
    let tools: Vec<Tool> = tools
        .unwrap_or_default()
        .into_iter()
        .flat_map(|tool| tool.function_declarations.unwrap_or_default())
        .map(|declaration: GeminiFunctionDeclaration| Tool {
            name: declaration.name,
            description: declaration.description,
            input_schema: declaration
                .parameters
                .unwrap_or_else(|| serde_json::json!({})),
        })
        .collect();

    if tools.is_empty() { None } else { Some(tools) }
}

fn push_gemini_part_stream_events(
    part: &GeminiPart,
    state: &mut GeminiStreamState,
    events: &mut Vec<StreamEvent>,
) {
    let index = state.next_index;

    if let Some(text) = &part.text {
        if !text.is_empty() {
            let start_block = if part.thought.unwrap_or(false) {
                ContentBlock::Thinking {
                    thinking: String::new(),
                    signature: None,
                }
            } else {
                ContentBlock::Text {
                    text: String::new(),
                    cache_control: None,
                }
            };
            events.push(StreamEvent::ContentBlockStart {
                index,
                content_block: start_block,
            });
            events.push(StreamEvent::ContentBlockDelta {
                index,
                delta: if part.thought.unwrap_or(false) {
                    crate::types::ContentBlockDelta::ThinkingDelta {
                        thinking: text.clone(),
                    }
                } else {
                    crate::types::ContentBlockDelta::TextDelta { text: text.clone() }
                },
            });
            if part.thought.unwrap_or(false) {
                if let Some(signature) = &part.thought_signature {
                    state.pending_thought_signature = Some(signature.clone());
                    events.push(StreamEvent::ContentBlockDelta {
                        index,
                        delta: crate::types::ContentBlockDelta::SignatureDelta {
                            signature: signature.clone(),
                        },
                    });
                }
            }
            events.push(StreamEvent::ContentBlockStop { index });
            state.next_index += 1;
        }
    }

    if let Some(function_call) = &part.function_call {
        let thought_signature = part
            .thought_signature
            .clone()
            .or_else(|| state.pending_thought_signature.take());
        events.push(StreamEvent::ContentBlockStart {
            index: state.next_index,
            content_block: ContentBlock::ToolUse {
                id: function_call
                    .id
                    .clone()
                    .unwrap_or_else(|| synthetic_tool_id(&function_call.name)),
                name: function_call.name.clone(),
                input: serde_json::json!({}),
                thought_signature,
                cache_control: None,
            },
        });
        events.push(StreamEvent::ContentBlockDelta {
            index: state.next_index,
            delta: crate::types::ContentBlockDelta::InputJsonDelta {
                partial_json: function_call.args.to_string(),
            },
        });
        events.push(StreamEvent::ContentBlockStop {
            index: state.next_index,
        });
        state.next_index += 1;
    }
}

fn convert_gemini_part_to_content_block(
    part: GeminiPart,
    pending_thought_signature: &mut Option<String>,
) -> Option<ContentBlock> {
    let cache_control = part.cache_control.clone();

    if let Some(text) = part.text {
        if part.thought.unwrap_or(false) {
            if let Some(signature) = &part.thought_signature {
                *pending_thought_signature = Some(signature.clone());
            }
        }
        return Some(ContentBlock::Text {
            text,
            cache_control,
        });
    }

    if let Some(GeminiInlineData { mime_type, data }) = part.inline_data {
        return Some(ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: mime_type,
                data,
            },
            cache_control,
        });
    }

    if let Some(GeminiFileData { file_uri, .. }) = part.file_data {
        return Some(ContentBlock::Image {
            source: ImageSource::Url { url: file_uri },
            cache_control,
        });
    }

    if let Some(GeminiFunctionCall { name, args, .. }) = part.function_call {
        return Some(ContentBlock::ToolUse {
            id: synthetic_tool_id(&name),
            name,
            input: args,
            thought_signature: part
                .thought_signature
                .or_else(|| pending_thought_signature.take()),
            cache_control,
        });
    }

    if let Some(GeminiFunctionResponse { name, response }) = part.function_response {
        return Some(ContentBlock::ToolResult {
            tool_use_id: synthetic_tool_id(&name),
            content: Some(ToolResultMessageContent::String(stringify_json_value(
                &response,
            ))),
            is_error: None,
            cache_control,
        });
    }

    None
}

fn convert_content_block_to_gemini_part(
    block: ContentBlock,
    tool_names: &mut HashMap<String, String>,
) -> Option<GeminiPart> {
    match block {
        ContentBlock::Text { text, .. } => Some(GeminiPart {
            text: Some(text),
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            cache_control: None,
        }),
        ContentBlock::Thinking {
            thinking,
            signature,
        } => Some(GeminiPart {
            text: Some(thinking),
            thought: Some(true),
            thought_signature: signature,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            cache_control: None,
        }),
        ContentBlock::Image { source, .. } => match source {
            ImageSource::Base64 { media_type, data } => Some(GeminiPart {
                text: None,
                thought: None,
                thought_signature: None,
                inline_data: Some(GeminiInlineData {
                    mime_type: media_type,
                    data,
                }),
                file_data: None,
                function_call: None,
                function_response: None,
                cache_control: None,
            }),
            ImageSource::Url { url } => Some(GeminiPart {
                text: None,
                thought: None,
                thought_signature: None,
                inline_data: None,
                file_data: Some(GeminiFileData {
                    mime_type: "application/octet-stream".to_string(),
                    file_uri: url,
                }),
                function_call: None,
                function_response: None,
                cache_control: None,
            }),
        },
        ContentBlock::ToolUse {
            id,
            name,
            input,
            thought_signature,
            ..
        } => {
            tool_names.insert(id.clone(), name.clone());
            debug!(
                id = %id,
                name = %name,
                has_thought_signature = thought_signature.is_some(),
                "Converting tool_use block to Gemini functionCall part"
            );
            Some(GeminiPart {
                text: None,
                thought: None,
                thought_signature,
                inline_data: None,
                file_data: None,
                function_call: Some(GeminiFunctionCall {
                    name,
                    args: input,
                    id: Some(id),
                }),
                function_response: None,
                cache_control: None,
            })
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
        } => {
            let name = tool_names
                .get(&tool_use_id)
                .cloned()
                .or_else(|| tool_use_id.strip_prefix("gemini-").map(ToString::to_string))
                .unwrap_or(tool_use_id);

            Some(GeminiPart {
                text: None,
                thought: None,
                thought_signature: None,
                inline_data: None,
                file_data: None,
                function_call: None,
                function_response: Some(GeminiFunctionResponse {
                    name,
                    response: convert_tool_result_content_to_gemini_response(content, is_error),
                }),
                cache_control: None,
            })
        }
        _ => None,
    }
}

fn convert_tool_result_content_to_gemini_response(
    content: Option<ToolResultMessageContent>,
    is_error: Option<bool>,
) -> Value {
    let content_value = match content {
        Some(ToolResultMessageContent::String(text)) => {
            serde_json::from_str::<Value>(&text).unwrap_or_else(|_| Value::String(text))
        }
        Some(ToolResultMessageContent::Blocks(blocks)) => Value::Array(
            blocks
                .into_iter()
                .map(|block| match block {
                    crate::types::ToolResultContentBlock::Text { text, .. } => {
                        serde_json::json!({
                            "type": "text",
                            "text": text,
                        })
                    }
                    crate::types::ToolResultContentBlock::Image { source, .. } => {
                        serde_json::json!({
                            "type": "image",
                            "source": source,
                        })
                    }
                })
                .collect(),
        ),
        None => Value::Null,
    };

    let mut response = match content_value {
        Value::Object(map) => Value::Object(map),
        other => serde_json::json!({
            "content": other,
        }),
    };

    if is_error == Some(true) {
        response = serde_json::json!({
            "is_error": true,
            "content": response,
        });
    }

    response
}

fn text_chunk(text: String) -> GeminiGenerateContentResponse {
    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: vec![GeminiPart {
                    text: Some(text),
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    cache_control: None,
                }],
            }),
            finish_reason: None,
            finish_message: None,
            index: Some(0),
        }]),
        usage_metadata: None,
        model_version: None,
        response_id: None,
    }
}

fn thought_chunk(text: String, signature: Option<String>) -> GeminiGenerateContentResponse {
    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: vec![GeminiPart {
                    text: Some(text),
                    thought: Some(true),
                    thought_signature: signature,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    cache_control: None,
                }],
            }),
            finish_reason: None,
            finish_message: None,
            index: Some(0),
        }]),
        usage_metadata: None,
        model_version: None,
        response_id: None,
    }
}

fn function_call_chunk(
    name: String,
    args: Value,
    id: Option<String>,
) -> GeminiGenerateContentResponse {
    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: vec![GeminiPart {
                    text: None,
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: Some(GeminiFunctionCall { name, args, id }),
                    function_response: None,
                    cache_control: None,
                }],
            }),
            finish_reason: None,
            finish_message: None,
            index: Some(0),
        }]),
        usage_metadata: None,
        model_version: None,
        response_id: None,
    }
}

fn error_chunk(error: ErrorDetails) -> GeminiGenerateContentResponse {
    GeminiGenerateContentResponse {
        candidates: Some(vec![GeminiCandidate {
            content: Some(GeminiContent {
                role: Some("model".to_string()),
                parts: vec![GeminiPart {
                    text: Some(error.message),
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    cache_control: None,
                }],
            }),
            finish_reason: Some("OTHER".to_string()),
            finish_message: None,
            index: Some(0),
        }]),
        usage_metadata: None,
        model_version: None,
        response_id: None,
    }
}

fn map_stop_reason(reason: Option<&str>) -> String {
    match reason {
        Some("max_tokens") => "MAX_TOKENS",
        Some("stop_sequence") => "STOP",
        Some("tool_use") => "STOP",
        Some("end_turn") | None => "STOP",
        Some(_) => "OTHER",
    }
    .to_string()
}

fn stringify_json_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

fn synthetic_tool_id(name: &str) -> String {
    format!("gemini-{}", name)
}

fn finish_message_for_stop_reason(reason: Option<&str>) -> Option<String> {
    match reason {
        Some("tool_use") => Some("Model generated function call(s).".to_string()),
        _ => None,
    }
}

fn map_gemini_finish_reason_to_stop_reason(reason: &str) -> String {
    match reason {
        "MAX_TOKENS" => "max_tokens",
        "STOP" => "end_turn",
        _ => "end_turn",
    }
    .to_string()
}

fn map_candidate_finish_reason_to_stop_reason(
    reason: &str,
    content: Option<&GeminiContent>,
) -> String {
    if reason == "STOP"
        && content
            .map(|content| {
                content
                    .parts
                    .iter()
                    .any(|part| part.function_call.is_some())
            })
            .unwrap_or(false)
    {
        return "tool_use".to_string();
    }

    map_gemini_finish_reason_to_stop_reason(reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ContentBlockDelta, GeminiGenerationConfig, MessageDelta, MessageDeltaUsage, Usage,
    };

    #[test]
    fn converts_gemini_request_to_anthropic() {
        let request = GeminiGenerateContentRequest {
            contents: vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![
                    GeminiPart {
                        text: Some("hi".to_string()),
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: None,
                        function_response: None,
                        cache_control: None,
                    },
                    GeminiPart {
                        text: None,
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: Some(GeminiFunctionCall {
                            name: "lookup".to_string(),
                            args: serde_json::json!({"ticker":"AAPL"}),
                            id: None,
                        }),
                        function_response: None,
                        cache_control: None,
                    },
                ],
            }],
            system_instruction: Some(GeminiContent {
                role: None,
                parts: vec![GeminiPart {
                    text: Some("be helpful".to_string()),
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    cache_control: None,
                }],
            }),
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: Some(321),
                temperature: Some(0.3),
                top_p: Some(0.8),
                stop_sequences: Some(vec!["DONE".to_string()]),
            }),
            tools: Some(vec![GeminiTool {
                function_declarations: Some(vec![GeminiFunctionDeclaration {
                    name: "lookup".to_string(),
                    description: Some("Find data".to_string()),
                    parameters: Some(serde_json::json!({"type":"object"})),
                }]),
            }]),
        };

        let anthropic = convert_gemini_to_anthropic("standard".to_string(), false, request);
        assert_eq!(anthropic.model, "standard");
        assert_eq!(anthropic.max_tokens, 321);
        assert_eq!(anthropic.stream, Some(false));
        assert_eq!(anthropic.messages.len(), 1);
        assert!(anthropic.system.is_some());
        assert_eq!(anthropic.tools.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn converts_anthropic_request_to_gemini() {
        let request = AnthropicRequest {
            model: "gemini-2.5-flash".to_string(),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Text {
                            text: "hello".to_string(),
                            cache_control: None,
                        },
                        ContentBlock::ToolResult {
                            tool_use_id: "tool-1".to_string(),
                            content: Some(ToolResultMessageContent::String(
                                "{\"ok\":true}".to_string(),
                            )),
                            is_error: None,
                            cache_control: None,
                        },
                    ]),
                },
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                        id: "tool-1".to_string(),
                        name: "lookup".to_string(),
                        input: serde_json::json!({"ticker":"AAPL"}),
                        thought_signature: None,
                        cache_control: None,
                    }]),
                },
            ],
            max_tokens: 222,
            metadata: None,
            stop_sequences: Some(vec!["END".to_string()]),
            stream: Some(true),
            system: Some(SystemPrompt::String("system".to_string())),
            temperature: Some(0.2),
            thinking: None,
            tool_choice: None,
            tools: Some(vec![Tool {
                name: "lookup".to_string(),
                description: Some("Look up a ticker".to_string()),
                input_schema: serde_json::json!({"type":"object"}),
            }]),
            top_k: None,
            top_p: Some(0.8),
            cache_control: None,
        };

        let gemini = convert_anthropic_to_gemini_request(request);
        assert_eq!(gemini.contents.len(), 2);
        assert_eq!(
            gemini
                .generation_config
                .as_ref()
                .and_then(|config| config.max_output_tokens),
            Some(222)
        );
        assert_eq!(
            gemini
                .system_instruction
                .as_ref()
                .and_then(|content| content.parts.first())
                .and_then(|part| part.text.as_deref()),
            Some("system")
        );
        assert_eq!(
            gemini
                .tools
                .as_ref()
                .and_then(|tools| tools.first())
                .and_then(|tool| tool.function_declarations.as_ref())
                .map(Vec::len),
            Some(1)
        );
        let function_response = gemini
            .contents
            .first()
            .and_then(|content| content.parts.get(1))
            .and_then(|part| part.function_response.as_ref());
        assert_eq!(
            function_response.map(|response| response.name.as_str()),
            Some("lookup")
        );
        assert_eq!(
            function_response.map(|response| response.response.clone()),
            Some(serde_json::json!({"ok":true}))
        );
    }

    #[test]
    fn wraps_plain_text_tool_result_for_gemini_function_response() {
        let request = AnthropicRequest {
            model: "gemini-2.5-flash".to_string(),
            messages: vec![
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                        id: "tool-1".to_string(),
                        name: "exec_command".to_string(),
                        input: serde_json::json!({"cmd":"ls -la"}),
                        thought_signature: None,
                        cache_control: None,
                    }]),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: "tool-1".to_string(),
                        content: Some(ToolResultMessageContent::String(
                            "plain text output".to_string(),
                        )),
                        is_error: None,
                        cache_control: None,
                    }]),
                },
            ],
            max_tokens: 128,
            metadata: None,
            stop_sequences: None,
            stream: Some(true),
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: Some(vec![Tool {
                name: "exec_command".to_string(),
                description: Some("Run a shell command".to_string()),
                input_schema: serde_json::json!({"type":"object"}),
            }]),
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        let gemini = convert_anthropic_to_gemini_request(request);
        let function_response = gemini
            .contents
            .get(1)
            .and_then(|content| content.parts.first())
            .and_then(|part| part.function_response.as_ref())
            .expect("expected function response");

        assert_eq!(
            function_response.response,
            serde_json::json!({"content":"plain text output"})
        );
    }

    #[test]
    fn wraps_error_text_tool_result_for_gemini_function_response() {
        let request = AnthropicRequest {
            model: "gemini-2.5-flash".to_string(),
            messages: vec![
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                        id: "tool-1".to_string(),
                        name: "exec_command".to_string(),
                        input: serde_json::json!({"cmd":"ls -la"}),
                        thought_signature: None,
                        cache_control: None,
                    }]),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: "tool-1".to_string(),
                        content: Some(ToolResultMessageContent::String(
                            "permission denied".to_string(),
                        )),
                        is_error: Some(true),
                        cache_control: None,
                    }]),
                },
            ],
            max_tokens: 128,
            metadata: None,
            stop_sequences: None,
            stream: Some(true),
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: Some(vec![Tool {
                name: "exec_command".to_string(),
                description: Some("Run a shell command".to_string()),
                input_schema: serde_json::json!({"type":"object"}),
            }]),
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        let gemini = convert_anthropic_to_gemini_request(request);
        let function_response = gemini
            .contents
            .get(1)
            .and_then(|content| content.parts.first())
            .and_then(|part| part.function_response.as_ref())
            .expect("expected function response");

        assert_eq!(
            function_response.response,
            serde_json::json!({
                "is_error": true,
                "content": {
                    "content": "permission denied"
                }
            })
        );
    }

    #[test]
    fn converts_anthropic_response_to_gemini() {
        let response = AnthropicResponse {
            id: "msg_1".to_string(),
            msg_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "hello".to_string(),
                    cache_control: None,
                },
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "lookup".to_string(),
                    input: serde_json::json!({"ticker":"AAPL"}),
                    thought_signature: None,
                    cache_control: None,
                },
            ],
            model: "x".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                cache_read_input_tokens: Some(4),
                output_tokens: 5,
                ..Default::default()
            },
        };

        let gemini = convert_anthropic_to_gemini(response);
        assert_eq!(gemini.candidates.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            gemini
                .usage_metadata
                .as_ref()
                .map(|usage| usage.total_token_count),
            Some(Some(15))
        );
        assert_eq!(
            gemini
                .usage_metadata
                .as_ref()
                .map(|usage| usage.cached_content_token_count),
            Some(Some(4))
        );
    }

    #[test]
    fn converts_gemini_response_to_anthropic() {
        let response = GeminiGenerateContentResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![
                        GeminiPart {
                            text: Some("hi".to_string()),
                            thought: None,
                            thought_signature: None,
                            inline_data: None,
                            file_data: None,
                            function_call: None,
                            function_response: None,
                            cache_control: None,
                        },
                        GeminiPart {
                            text: None,
                            thought: None,
                            thought_signature: None,
                            inline_data: None,
                            file_data: None,
                            function_call: Some(GeminiFunctionCall {
                                name: "lookup".to_string(),
                                args: serde_json::json!({"ticker":"AAPL"}),
                                id: Some("call-1".to_string()),
                            }),
                            function_response: None,
                            cache_control: None,
                        },
                    ],
                }),
                finish_reason: Some("STOP".to_string()),
                finish_message: None,
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                cached_content_token_count: Some(4),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
            }),
            model_version: Some("gemini-2.5-flash".to_string()),
            response_id: Some("resp-1".to_string()),
        };

        let anthropic = convert_gemini_response_to_anthropic("fallback", response);
        assert_eq!(anthropic.id, "resp-1");
        assert_eq!(anthropic.model, "gemini-2.5-flash");
        assert_eq!(anthropic.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(anthropic.usage.input_tokens, 10);
        assert_eq!(anthropic.usage.cache_read_input_tokens, Some(4));
        assert_eq!(anthropic.usage.output_tokens, 5);
        assert_eq!(anthropic.content.len(), 2);
    }

    #[test]
    fn converts_gemini_stream_request_to_anthropic_with_stream_flag() {
        let request = GeminiGenerateContentRequest {
            contents: vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart {
                    text: Some("hi".to_string()),
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    cache_control: None,
                }],
            }],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };

        let anthropic = convert_gemini_to_anthropic("standard".to_string(), true, request);
        assert_eq!(anthropic.stream, Some(true));
    }

    #[test]
    fn converts_gemini_stream_chunk_to_anthropic_events() {
        let mut state = GeminiStreamState::default();
        let chunk = GeminiGenerateContentResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart {
                        text: Some("hello".to_string()),
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: None,
                        function_response: None,
                        cache_control: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
                finish_message: None,
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(8),
                cached_content_token_count: Some(2),
                candidates_token_count: Some(3),
                total_token_count: Some(11),
            }),
            model_version: Some("gemini-2.5-flash".to_string()),
            response_id: Some("resp-1".to_string()),
        };

        let events = convert_gemini_stream_chunk_to_anthropic("fallback", chunk, &mut state);
        assert!(matches!(
            events.first(),
            Some(StreamEvent::MessageStart { .. })
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockDelta {
                delta: ContentBlockDelta::TextDelta { text },
                ..
            } if text == "hello"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageDelta {
                delta: MessageDelta { stop_reason, .. },
                usage: MessageDeltaUsage {
                    output_tokens: 3,
                    input_tokens: Some(8),
                    cache_read_input_tokens: Some(2),
                    ..
                },
            } if stop_reason.as_deref() == Some("end_turn")
        )));
        assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
    }

    #[test]
    fn maps_streaming_gemini_tool_calls_to_tool_use_stop_reason() {
        let mut state = GeminiStreamState::default();
        let chunk = GeminiGenerateContentResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart {
                        text: Some(String::new()),
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: Some(GeminiFunctionCall {
                            name: "exec_command".to_string(),
                            args: serde_json::json!({"cmd":"ls -la"}),
                            id: Some("call-1".to_string()),
                        }),
                        function_response: None,
                        cache_control: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
                finish_message: None,
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                cached_content_token_count: Some(2),
                candidates_token_count: Some(0),
                total_token_count: Some(10),
            }),
            model_version: Some("gemini-3.1-pro-preview".to_string()),
            response_id: Some("resp-1".to_string()),
        };

        let events = convert_gemini_stream_chunk_to_anthropic("fallback", chunk, &mut state);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse { name, .. },
                ..
            } if name == "exec_command"
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockDelta {
                delta: ContentBlockDelta::TextDelta { text },
                ..
            } if text.is_empty()
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageDelta {
                delta: MessageDelta { stop_reason, .. },
                ..
            } if stop_reason.as_deref() == Some("tool_use")
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageStop))
        );
    }

    #[test]
    fn carries_thought_signature_from_thought_part_to_following_tool_call() {
        let mut state = GeminiStreamState::default();
        let chunk = GeminiGenerateContentResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![
                        GeminiPart {
                            text: Some("thinking".to_string()),
                            thought: Some(true),
                            thought_signature: Some("sig-123".to_string()),
                            inline_data: None,
                            file_data: None,
                            function_call: None,
                            function_response: None,
                            cache_control: None,
                        },
                        GeminiPart {
                            text: None,
                            thought: None,
                            thought_signature: None,
                            inline_data: None,
                            file_data: None,
                            function_call: Some(GeminiFunctionCall {
                                name: "exec_command".to_string(),
                                args: serde_json::json!({"cmd":"ls -la"}),
                                id: Some("call-1".to_string()),
                            }),
                            function_response: None,
                            cache_control: None,
                        },
                    ],
                }),
                finish_reason: Some("STOP".to_string()),
                finish_message: None,
                index: Some(0),
            }]),
            usage_metadata: None,
            model_version: Some("gemini-3.1-pro-preview".to_string()),
            response_id: Some("resp-1".to_string()),
        };

        let events = convert_gemini_stream_chunk_to_anthropic("fallback", chunk, &mut state);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse {
                    thought_signature,
                    ..
                },
                ..
            } if thought_signature.as_deref() == Some("sig-123")
        )));
    }

    #[test]
    fn maps_non_streaming_gemini_tool_calls_to_tool_use_stop_reason() {
        let response = GeminiGenerateContentResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart {
                        text: None,
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: Some(GeminiFunctionCall {
                            name: "exec_command".to_string(),
                            args: serde_json::json!({"cmd":"ls -la"}),
                            id: Some("call-1".to_string()),
                        }),
                        function_response: None,
                        cache_control: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
                finish_message: None,
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                cached_content_token_count: None,
                candidates_token_count: Some(0),
                total_token_count: Some(10),
            }),
            model_version: Some("gemini-3.1-pro-preview".to_string()),
            response_id: Some("resp-1".to_string()),
        };

        let anthropic = convert_gemini_response_to_anthropic("fallback", response);
        assert_eq!(anthropic.stop_reason.as_deref(), Some("tool_use"));
    }

    #[test]
    fn preserves_cache_control_when_converting_gemini_to_anthropic() {
        let request = GeminiGenerateContentRequest {
            contents: vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart {
                    text: Some("cache me".to_string()),
                    thought: None,
                    thought_signature: None,
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    cache_control: Some(crate::types::CacheControl {
                        control_type: "ephemeral".to_string(),
                        ttl: None,
                    }),
                }],
            }],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };

        let anthropic = convert_gemini_to_anthropic("standard".to_string(), false, request);
        let cache_control = anthropic
            .messages
            .first()
            .and_then(|message| match &message.content {
                MessageContent::Blocks(blocks) => blocks.first(),
                MessageContent::String(_) => None,
            })
            .and_then(|block| match block {
                ContentBlock::Text { cache_control, .. } => cache_control.as_ref(),
                _ => None,
            });

        assert_eq!(
            cache_control.map(|value| value.control_type.as_str()),
            Some("ephemeral")
        );
    }

    #[test]
    fn converts_stream_delta_to_gemini_text_chunk() {
        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "hel".to_string(),
            },
        };
        let mut response_id = None;
        let mut model_version = None;
        let mut pending_tool_json = String::new();
        let mut pending_tool_name = None;
        let mut pending_tool_id = None;

        let response = convert_stream_event_to_gemini(
            event,
            &mut response_id,
            &mut model_version,
            &mut pending_tool_json,
            &mut pending_tool_name,
            &mut pending_tool_id,
        )
        .expect("chunk");

        let text = response
            .candidates
            .as_ref()
            .and_then(|candidates| candidates.first())
            .and_then(|candidate| candidate.content.as_ref())
            .and_then(|content| content.parts.first())
            .and_then(|part| part.text.as_ref())
            .cloned();
        assert_eq!(text.as_deref(), Some("hel"));
    }

    #[test]
    fn converts_message_delta_to_usage_chunk() {
        let event = StreamEvent::MessageDelta {
            delta: MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
            },
            usage: MessageDeltaUsage {
                output_tokens: 6,
                input_tokens: Some(4),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: Some(2),
            },
        };
        let mut response_id = None;
        let mut model_version = None;
        let mut pending_tool_json = String::new();
        let mut pending_tool_name = None;
        let mut pending_tool_id = None;

        let response = convert_stream_event_to_gemini(
            event,
            &mut response_id,
            &mut model_version,
            &mut pending_tool_json,
            &mut pending_tool_name,
            &mut pending_tool_id,
        )
        .expect("chunk");

        assert_eq!(
            response
                .usage_metadata
                .as_ref()
                .map(|usage| usage.total_token_count),
            Some(Some(10))
        );
        assert_eq!(
            response
                .usage_metadata
                .as_ref()
                .map(|usage| usage.cached_content_token_count),
            Some(Some(2))
        );
    }

    #[test]
    fn converts_partial_gemini_usage_metadata_without_parse_failures() {
        let mut state = GeminiStreamState::default();
        let chunk = GeminiGenerateContentResponse {
            candidates: None,
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(4),
                cached_content_token_count: Some(1),
                candidates_token_count: None,
                total_token_count: Some(4),
            }),
            model_version: Some("gemini-3.1-pro".to_string()),
            response_id: Some("resp-1".to_string()),
        };

        let events = convert_gemini_stream_chunk_to_anthropic("fallback", chunk, &mut state);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageDelta {
                usage: MessageDeltaUsage {
                    output_tokens: 0,
                    input_tokens: Some(4),
                    cache_read_input_tokens: Some(1),
                    ..
                },
                ..
            }
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageStop))
        );
    }

    #[test]
    fn does_not_stop_on_non_terminal_gemini_chunk_with_usage_metadata() {
        let mut state = GeminiStreamState::default();
        let chunk = GeminiGenerateContentResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart {
                        text: Some("I am a coding agent running in".to_string()),
                        thought: None,
                        thought_signature: None,
                        inline_data: None,
                        file_data: None,
                        function_call: None,
                        function_response: None,
                        cache_control: None,
                    }],
                }),
                finish_reason: None,
                finish_message: None,
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10862),
                cached_content_token_count: None,
                candidates_token_count: Some(7),
                total_token_count: Some(10869),
            }),
            model_version: Some("gemini-3.1-pro-preview".to_string()),
            response_id: Some("resp-1".to_string()),
        };

        let events = convert_gemini_stream_chunk_to_anthropic("fallback", chunk, &mut state);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockDelta {
                delta: ContentBlockDelta::TextDelta { text },
                ..
            } if text == "I am a coding agent running in"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason: None,
                    ..
                },
                usage: MessageDeltaUsage {
                    output_tokens: 7,
                    input_tokens: Some(10862),
                    ..
                },
            }
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageStop))
        );
    }

    #[test]
    fn buffers_tool_json_only_from_deltas() {
        let mut response_id = None;
        let mut model_version = None;
        let mut pending_tool_json = String::new();
        let mut pending_tool_name = None;
        let mut pending_tool_id = None;

        let start = StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "run_shell_command".to_string(),
                input: serde_json::json!({}),
                thought_signature: None,
                cache_control: None,
            },
        };

        assert!(
            convert_stream_event_to_gemini(
                start,
                &mut response_id,
                &mut model_version,
                &mut pending_tool_json,
                &mut pending_tool_name,
                &mut pending_tool_id,
            )
            .is_none()
        );

        let delta = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::InputJsonDelta {
                partial_json: "{\"command\":\"echo hi\"}".to_string(),
            },
        };

        assert!(
            convert_stream_event_to_gemini(
                delta,
                &mut response_id,
                &mut model_version,
                &mut pending_tool_json,
                &mut pending_tool_name,
                &mut pending_tool_id,
            )
            .is_none()
        );

        let stop = StreamEvent::ContentBlockStop { index: 0 };
        let response = convert_stream_event_to_gemini(
            stop,
            &mut response_id,
            &mut model_version,
            &mut pending_tool_json,
            &mut pending_tool_name,
            &mut pending_tool_id,
        )
        .expect("tool chunk");

        let args = response
            .candidates
            .as_ref()
            .and_then(|candidates| candidates.first())
            .and_then(|candidate| candidate.content.as_ref())
            .and_then(|content| content.parts.first())
            .and_then(|part| part.function_call.as_ref())
            .map(|function_call| function_call.args.clone());

        assert_eq!(args, Some(serde_json::json!({"command":"echo hi"})));
    }
}
