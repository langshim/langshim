use crate::adapter::ModelAdapter;
use crate::models::Provider;
use crate::types::{
    AnthropicRequest, AnthropicResponse, ContentBlock, ContentBlockDelta, Message, MessageContent,
    MessageDelta, MessageDeltaUsage, StreamEvent, SystemPrompt, ToolChoice, Usage,
};
use async_openai::types::chat::{
    ChatCompletionNamedToolChoice, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    FunctionName, FunctionObject, ImageUrl, ReasoningEffort, ToolChoiceOptions,
};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::json;
use std::error::Error;
use tracing::debug;

#[derive(Clone)]
pub struct OpenAIAdapter {
    http_client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAIAdapter {
    pub fn default_converter() -> Self {
        Self::new(String::new(), None)
    }

    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let base_url_str = if let Some(url) = base_url {
            url
        } else {
            "https://api.openai.com/v1".to_string()
        };
        let http_client = reqwest::Client::new();
        Self {
            http_client,
            api_key,
            base_url: base_url_str,
        }
    }

    fn is_openrouter(&self) -> bool {
        self.base_url.contains("openrouter.ai")
    }

    fn parse_data_url_image(url: &str) -> Option<crate::types::ImageSource> {
        let data = url.strip_prefix("data:")?;
        let (header, payload) = data.split_once(',')?;
        let media_type = header.strip_suffix(";base64")?;

        Some(crate::types::ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: payload.to_string(),
        })
    }

    fn apply_openrouter_request_fields(
        &self,
        request_value: &mut serde_json::Value,
        request: &AnthropicRequest,
        openai_model: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        if !self.is_openrouter() {
            return Ok(());
        }

        let Some(map) = request_value.as_object_mut() else {
            return Ok(());
        };

        map.insert("cache_control".to_string(), json!({"type": "ephemeral"}));

        if let Some(thinking) = &request.thinking {
            map.insert("thinking".to_string(), serde_json::to_value(thinking)?);

            if openai_model.starts_with("anthropic/") {
                if thinking.thinking_type == "enabled" {
                    if let Some(budget) = thinking.budget_tokens {
                        if budget > 0 {
                            map.insert("reasoning".to_string(), json!({"max_tokens": budget}));
                        }
                    }
                } else if thinking.thinking_type == "adaptive" {
                    map.insert("reasoning".to_string(), json!({"max_tokens": 32000}));
                }
            } else if openai_model.starts_with("openai/")
                && (thinking.thinking_type == "enabled" || thinking.thinking_type == "adaptive")
            {
                map.insert("reasoning".to_string(), json!({"effort": "medium"}));
            }
        }

        Ok(())
    }

    fn sanitize_provider_request_value(request_value: &mut serde_json::Value, is_stream: bool) {
        if let Some(map) = request_value.as_object_mut() {
            map.remove("reasoning_effort");
            if !is_stream {
                map.remove("stream_options");
            }
        }
    }

    fn convert_request(
        &self,
        request: &AnthropicRequest,
    ) -> Result<CreateChatCompletionRequest, Box<dyn Error + Send + Sync>> {
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

        // 1. Handle System Prompt
        if let Some(system) = &request.system {
            let content = match system {
                SystemPrompt::String(s) => s.clone(),
                SystemPrompt::Blocks(blocks) => {
                    blocks
                        .iter()
                        .map(|b| match b {
                            ContentBlock::Text { text, .. } => text.clone(),
                            _ => "".to_string(), // OpenAI system prompt is text-only usually
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            };
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(content),
                    name: None,
                },
            ));
        }

        // 2. Handle Messages
        for msg in &request.messages {
            match msg.role.as_str() {
                "user" => {
                    let content_parts = match &msg.content {
                        MessageContent::String(s) => {
                            vec![ChatCompletionRequestUserMessageContentPart::Text(
                                ChatCompletionRequestMessageContentPartTextArgs::default()
                                    .text(s)
                                    .build()?,
                            )]
                        }
                        MessageContent::Blocks(blocks) => {
                            let mut parts = Vec::new();
                            for block in blocks {
                                match block {
                                    ContentBlock::Text { text, .. } => {
                                        parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                                            ChatCompletionRequestMessageContentPartTextArgs::default().text(text).build()?
                                        ));
                                    }
                                    ContentBlock::Image { source, .. } => {
                                        let url = match source {
                                            crate::types::ImageSource::Base64 {
                                                media_type,
                                                data,
                                            } => {
                                                format!("data:{};base64,{}", media_type, data)
                                            }
                                            crate::types::ImageSource::Url { url } => url.clone(),
                                        };
                                        parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                                            ChatCompletionRequestMessageContentPartImageArgs::default()
                                                .image_url(ImageUrl {
                                                    url,
                                                    detail: None,
                                                })
                                                .build()?
                                        ));
                                    }
                                    _ => {} // Tool use/result not expected in user message usually, but possible in some flows.
                                            // For simplicity, ignoring tool_use/result in user messages for now as they map to assistant/tool roles.
                                }
                            }
                            parts
                        }
                    };
                    let content = if content_parts.len() == 1 {
                        if let ChatCompletionRequestUserMessageContentPart::Text(t) =
                            &content_parts[0]
                        {
                            ChatCompletionRequestUserMessageContent::Text(t.text.clone())
                        } else {
                            ChatCompletionRequestUserMessageContent::Array(content_parts)
                        }
                    } else {
                        ChatCompletionRequestUserMessageContent::Array(content_parts)
                    };
                    messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessage {
                            content,
                            name: None,
                        },
                    ));
                }
                "assistant" => {
                    // Handle assistant messages (could be text or tool calls)
                    // This is complex if we need to reconstruct tool calls from blocks
                    // For now, simplified text handling
                    let content = match &msg.content {
                        MessageContent::String(s) => {
                            ChatCompletionRequestAssistantMessageContent::Text(s.clone())
                        }
                        MessageContent::Blocks(blocks) => {
                            let text: String = blocks
                                .iter()
                                .filter_map(|b| match b {
                                    ContentBlock::Text { text, .. } => Some(text.clone()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("");
                            ChatCompletionRequestAssistantMessageContent::Text(text)
                        }
                    };
                    messages.push(ChatCompletionRequestMessage::Assistant(
                        ChatCompletionRequestAssistantMessage {
                            content: Some(content),
                            refusal: None,
                            name: None,
                            audio: None,
                            tool_calls: None,
                            ..Default::default()
                        },
                    ));
                }
                _ => {}
            }
        }

        // 3. Handle Tools
        let tools = if let Some(anthropic_tools) = &request.tools {
            Some(
                anthropic_tools
                    .iter()
                    .map(|t| {
                        ChatCompletionTools::Function(ChatCompletionTool {
                            function: FunctionObject {
                                name: t.name.clone(),
                                description: t.description.clone(),
                                parameters: Some(t.input_schema.clone()),
                                strict: None,
                            },
                        })
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // 4. Handle Tool Choice
        let tool_choice: Option<ChatCompletionToolChoiceOption> =
            if let Some(choice) = &request.tool_choice {
                match choice {
                    ToolChoice::Auto => Some(ChatCompletionToolChoiceOption::Mode(
                        ToolChoiceOptions::Auto,
                    )),
                    ToolChoice::Any => Some(ChatCompletionToolChoiceOption::Mode(
                        ToolChoiceOptions::Required,
                    )),
                    ToolChoice::Tool { name } => Some(ChatCompletionToolChoiceOption::Function(
                        ChatCompletionNamedToolChoice {
                            function: FunctionName { name: name.clone() },
                        },
                    )),
                }
            } else {
                None
            };

        //let model = request.model.strip_prefix("openai/").unwrap_or(&request.model);

        let max_tokens = request.max_tokens
            + request
                .thinking
                .as_ref()
                .and_then(|t| t.budget_tokens)
                .filter(|budget| *budget > 0)
                .map(|budget| budget as u32)
                .unwrap_or(0);

        let tools_vec = tools.unwrap_or_default();
        let mut binding = CreateChatCompletionRequestArgs::default();
        let mut req_builder = binding
            .stream_options(ChatCompletionStreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            })
            .model(&request.model)
            .messages(messages)
            .max_completion_tokens(max_tokens)
            .temperature(request.temperature.unwrap_or(1.0))
            // .top_p(request.top_p.unwrap_or(1.0))
            .tools(tools_vec);

        if let Some(tc) = tool_choice {
            req_builder = req_builder.tool_choice(tc);
        }

        Ok(req_builder.build()?)
    }

    fn convert_response_json(
        &self,
        response: &serde_json::Value,
        model: String,
    ) -> Result<AnthropicResponse, Box<dyn Error + Send + Sync>> {
        let choice = response
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first())
            .ok_or("missing choices[0] in OpenAI response")?;

        let content = if let Some(content) = choice
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(|content| content.as_str())
        {
            vec![ContentBlock::Text {
                text: content.to_string(),
                cache_control: None,
            }]
        } else if let Some(tool_calls) = choice
            .get("message")
            .and_then(|message| message.get("tool_calls"))
            .and_then(|tool_calls| tool_calls.as_array())
        {
            tool_calls
                .iter()
                .filter_map(|tc| {
                    let function = tc.get("function")?;
                    Some(ContentBlock::ToolUse {
                        id: tc.get("id")?.as_str()?.to_string(),
                        name: function.get("name")?.as_str()?.to_string(),
                        input: function
                            .get("arguments")
                            .and_then(|arguments| arguments.as_str())
                            .and_then(|arguments| serde_json::from_str(arguments).ok())
                            .unwrap_or(json!({})),
                        thought_signature: None,
                        cache_control: None,
                    })
                })
                .collect()
        } else {
            vec![]
        };

        let finish_reason = choice
            .get("finish_reason")
            .and_then(|reason| reason.as_str());
        let usage = response.get("usage");

        Ok(AnthropicResponse {
            id: response
                .get("id")
                .and_then(|id| id.as_str())
                .unwrap_or("msg_openai_placeholder")
                .to_string(),
            msg_type: "message".to_string(),
            role: "assistant".to_string(),
            content,
            model,
            stop_reason: Some(match finish_reason {
                Some("stop") => "end_turn".to_string(),
                Some("length") => "max_tokens".to_string(),
                Some("tool_calls") => "tool_use".to_string(),
                _ => "end_turn".to_string(),
            }),
            stop_sequence: None,
            usage: Usage {
                input_tokens: usage
                    .and_then(|usage| usage.get("prompt_tokens"))
                    .and_then(|tokens| tokens.as_u64())
                    .unwrap_or(0) as u32,
                cache_read_input_tokens: usage
                    .and_then(|usage| usage.get("prompt_tokens_details"))
                    .and_then(|details| details.get("cached_tokens"))
                    .and_then(|tokens| tokens.as_u64())
                    .map(|tokens| tokens as u32),
                cache_creation_input_tokens: usage
                    .and_then(|usage| usage.get("prompt_tokens_details"))
                    .and_then(|details| details.get("cache_write_tokens"))
                    .and_then(|tokens| tokens.as_u64())
                    .map(|tokens| tokens as u32),
                output_tokens: usage
                    .and_then(|usage| usage.get("completion_tokens"))
                    .and_then(|tokens| tokens.as_u64())
                    .unwrap_or(0) as u32,
            },
        })
    }

    /// Convert OpenAI ChatCompletionRequest to AnthropicRequest
    pub fn convert_to_anthropic(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<AnthropicRequest, Box<dyn Error + Send + Sync>> {
        let mut messages: Vec<Message> = Vec::new();
        let mut system: Option<SystemPrompt> = None;

        for msg in &request.messages {
            match msg {
                ChatCompletionRequestMessage::System(sys_msg) => {
                    let content = match &sys_msg.content {
                        ChatCompletionRequestSystemMessageContent::Text(text) => text.clone(),
                        ChatCompletionRequestSystemMessageContent::Array(_) => {
                            // Fallback: serialize to string
                            serde_json::to_string(&sys_msg.content)?
                        }
                    };
                    system = Some(SystemPrompt::String(content));
                }
                ChatCompletionRequestMessage::Developer(_) => {
                    // Developer messages are not supported in Anthropic API
                }
                ChatCompletionRequestMessage::User(user_msg) => {
                    let content = match &user_msg.content {
                        ChatCompletionRequestUserMessageContent::Text(text) => {
                            MessageContent::String(text.clone())
                        }
                        ChatCompletionRequestUserMessageContent::Array(parts) => {
                            let blocks: Vec<ContentBlock> = parts
                                .iter()
                                .filter_map(|part| match part {
                                    ChatCompletionRequestUserMessageContentPart::Text(t) => {
                                        Some(ContentBlock::Text {
                                            text: t.text.clone(),
                                            cache_control: None,
                                        })
                                    }
                                    ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                                        Some(ContentBlock::Image {
                                            source: Self::parse_data_url_image(&img.image_url.url)
                                                .unwrap_or_else(|| {
                                                    crate::types::ImageSource::Url {
                                                        url: img.image_url.url.clone(),
                                                    }
                                                }),
                                            cache_control: None,
                                        })
                                    }
                                    ChatCompletionRequestUserMessageContentPart::InputAudio(_) => {
                                        None
                                    }
                                    ChatCompletionRequestUserMessageContentPart::File(_) => None,
                                })
                                .collect();
                            MessageContent::Blocks(blocks)
                        }
                    };
                    messages.push(Message {
                        role: "user".to_string(),
                        content,
                    });
                }
                ChatCompletionRequestMessage::Assistant(assistant_msg) => {
                    let content = if let Some(text) = &assistant_msg.content {
                        match text {
                            ChatCompletionRequestAssistantMessageContent::Text(text) => {
                                MessageContent::String(text.clone())
                            }
                            ChatCompletionRequestAssistantMessageContent::Array(_) => {
                                MessageContent::String("".to_string())
                            }
                        }
                    } else {
                        MessageContent::String("".to_string())
                    };
                    messages.push(Message {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                ChatCompletionRequestMessage::Tool(_) => {
                    // Tool messages are not standard in OpenAI chat completions
                    // They typically appear as assistant messages with tool_calls
                }
                ChatCompletionRequestMessage::Function(_) => {
                    // Legacy function calling format
                }
            }
        }

        // Convert tools
        let tools = request.tools.map(|tools_vec| {
            tools_vec
                .into_iter()
                .filter_map(|t| {
                    if let ChatCompletionTools::Function(f) = t {
                        Some(crate::types::Tool {
                            name: f.function.name,
                            description: f.function.description,
                            input_schema: f.function.parameters.unwrap_or(serde_json::json!({})),
                        })
                    } else {
                        None
                    }
                })
                .collect()
        });

        // Convert tool_choice
        let tool_choice = request.tool_choice.map(|tc| match tc {
            ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto) => ToolChoice::Auto,
            ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Required) => ToolChoice::Any,
            ChatCompletionToolChoiceOption::Function(f) => ToolChoice::Tool {
                name: f.function.name,
            },
            _ => ToolChoice::Auto,
        });

        let model = request.model.clone();
        let max_tokens = request.max_tokens.unwrap_or(1024);

        // Set thinking budget based on reasoning_effort
        let thinking = match request.reasoning_effort {
            Some(ReasoningEffort::Minimal) => Some(crate::types::Thinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: Some(4000),
            }),
            Some(ReasoningEffort::Low) => Some(crate::types::Thinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: Some(8000),
            }),
            Some(ReasoningEffort::Medium) => Some(crate::types::Thinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: Some(16000),
            }),
            Some(ReasoningEffort::High) => Some(crate::types::Thinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: Some(32000),
            }),
            Some(ReasoningEffort::Xhigh) => Some(crate::types::Thinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: Some(64000),
            }),
            _ => None,
        };

        Ok(AnthropicRequest {
            model,
            messages,
            max_tokens,
            metadata: None,
            stop_sequences: None,
            stream: Some(request.stream.unwrap_or(false)),
            system,
            temperature: request.temperature,
            thinking,
            tool_choice,
            tools,
            top_k: None,
            top_p: request.top_p,
            cache_control: None,
        })
    }

    /// Convert AnthropicResponse to OpenAI ChatCompletion response JSON
    pub fn convert_response_to_openai(&self, response: AnthropicResponse) -> serde_json::Value {
        let content = response
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text, .. } => text.clone(),
                ContentBlock::ToolUse { name, input, .. } => serde_json::json!({
                    "tool_use": {
                        "name": name,
                        "input": input
                    }
                })
                .to_string(),
                _ => "".to_string(),
            })
            .collect::<Vec<_>>()
            .join("");

        let finish_reason = response.stop_reason.as_ref().map(|s| match s.as_str() {
            "end_turn" => "stop",
            "max_tokens" => "length",
            "tool_use" => "tool_calls",
            _ => "stop",
        });

        serde_json::json!({
            "id": response.id,
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": response.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        })
    }

    fn print_curl(&self, url: &str, body: &serde_json::Value) {
        let body_str = serde_json::to_string(body).unwrap_or_default();
        debug!("Generated cURL command:");
        debug!("curl -X POST \"{}\" \\", url);
        debug!("  -H \"Authorization: Bearer {}\" \\", self.api_key);
        debug!("  -H \"Content-Type: application/json\" \\");
        debug!("  -d '{}'", body_str.replace("'", "'\\''"));
    }
}

#[cfg(test)]
mod tests {
    use super::OpenAIAdapter;
    use crate::types::MessageContent;
    use async_openai::types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImageArgs,
        ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        CreateChatCompletionRequestArgs, ImageUrl,
    };
    use serde_json::json;

    #[test]
    fn convert_response_json_preserves_prompt_token_details() {
        let adapter = OpenAIAdapter::default_converter();
        let response = json!({
            "id": "chatcmpl_test",
            "model": "anthropic/claude-sonnet-4.6",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "ok"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12230,
                "completion_tokens": 110,
                "prompt_tokens_details": {
                    "cached_tokens": 4096,
                    "cache_write_tokens": 2048
                }
            }
        });

        let anthropic = adapter
            .convert_response_json(&response, "anthropic/claude-sonnet-4.6".to_string())
            .expect("response conversion should succeed");

        assert_eq!(anthropic.usage.input_tokens, 12230);
        assert_eq!(anthropic.usage.output_tokens, 110);
        assert_eq!(anthropic.usage.cache_read_input_tokens, Some(4096));
        assert_eq!(anthropic.usage.cache_creation_input_tokens, Some(2048));
    }

    #[test]
    fn convert_request_preserves_chat_data_url_images_as_base64() {
        let adapter = OpenAIAdapter::default_converter();
        let request = CreateChatCompletionRequestArgs::default()
            .model("anthropic/claude-sonnet-4.5")
            .max_completion_tokens(256u32)
            .messages(vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestUserMessageContentPart::Text(
                            ChatCompletionRequestMessageContentPartTextArgs::default()
                                .text("look")
                                .build()
                                .expect("text part should build"),
                        ),
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(
                            ChatCompletionRequestMessageContentPartImageArgs::default()
                                .image_url(ImageUrl {
                                    url: "data:image/png;base64,Zm9vYmFy".to_string(),
                                    detail: None,
                                })
                                .build()
                                .expect("image part should build"),
                        ),
                    ]),
                    name: None,
                },
            )])
            .build()
            .expect("request should build");

        let anthropic = adapter
            .convert_to_anthropic(request)
            .expect("request conversion should succeed");

        let MessageContent::Blocks(blocks) = &anthropic.messages[0].content else {
            panic!("expected block content");
        };

        let crate::types::ContentBlock::Image { source, .. } = &blocks[1] else {
            panic!("expected image block");
        };

        match source {
            crate::types::ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "Zm9vYmFy");
            }
            other => panic!("expected base64 source, got {other:?}"),
        }
    }

    #[test]
    fn sanitize_provider_request_value_removes_reasoning_effort_and_non_stream_options() {
        let mut request_value = json!({
            "model": "gpt-5.4-nano",
            "reasoning_effort": "medium",
            "stream_options": { "include_usage": true }
        });

        OpenAIAdapter::sanitize_provider_request_value(&mut request_value, false);

        assert!(request_value.get("reasoning_effort").is_none());
        assert!(request_value.get("stream_options").is_none());
    }

    #[test]
    fn sanitize_provider_request_value_keeps_stream_options_for_streaming() {
        let mut request_value = json!({
            "model": "gpt-5.4-nano",
            "reasoning_effort": "medium",
            "stream_options": { "include_usage": true }
        });

        OpenAIAdapter::sanitize_provider_request_value(&mut request_value, true);

        assert!(request_value.get("reasoning_effort").is_none());
        assert_eq!(
            request_value
                .get("stream_options")
                .and_then(|value| value.get("include_usage"))
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }
}

#[async_trait]
impl ModelAdapter for OpenAIAdapter {
    async fn send_message(
        &self,
        request: AnthropicRequest,
    ) -> Result<AnthropicResponse, Box<dyn Error + Send + Sync>> {
        let openai_request = self.convert_request(&request)?;
        let mut request_value = serde_json::to_value(&openai_request)?;
        self.apply_openrouter_request_fields(&mut request_value, &request, &openai_request.model)?;
        Self::sanitize_provider_request_value(&mut request_value, false);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        self.print_curl(&url, &request_value);

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_value)
            .send()
            .await?;

        if !response.status().is_success() {
            let text = response.text().await?;
            return Err(format!("OpenAI API Error: {}", text).into());
        }

        let response_json: serde_json::Value = response.json().await?;
        self.convert_response_json(&response_json, request.model)
    }

    async fn send_message_stream(
        &self,
        request: AnthropicRequest,
        provider: Provider,
    ) -> Result<
        BoxStream<'static, Result<StreamEvent, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    > {
        debug!("Anthropic Request: {:?}", request);
        let openai_request = self.convert_request(&request)?;
        debug!("OpenAI Request: {:?}", openai_request);
        let mut request_value = serde_json::to_value(&openai_request)?;
        self.apply_openrouter_request_fields(&mut request_value, &request, &openai_request.model)?;
        Self::sanitize_provider_request_value(&mut request_value, true);
        if let Some(map) = request_value.as_object_mut() {
            map.insert("stream".to_string(), json!(true));
            map.insert(
                "stream_options".to_string(),
                json!({ "include_usage": true }),
            );
            let mut total_max_tokens = request.max_tokens;
            map.insert("max_completion_tokens".to_string(), json!(total_max_tokens));
            // map.remove("max_completion_tokens");
            if provider == Provider::OpenRouter {
                if let Some(thinking) = &request.thinking {
                    if let Some(budget) = thinking.budget_tokens {
                        if budget > 0 {
                            total_max_tokens += budget as u32;
                        }
                    }
                    map.insert("max_completion_tokens".to_string(), json!(total_max_tokens));
                }
            }
        }

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        self.print_curl(&url, &request_value);

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_value)
            .send()
            .await?;

        if !response.status().is_success() {
            let text = response.text().await?;
            return Err(format!("OpenAI API Error: {}", text).into());
        }

        let mut stream = response.bytes_stream();

        // let input_tokens = self.count_tokens(request.clone()).await.unwrap_or(0);

        let output_stream = async_stream::try_stream! {
            let mut buffer = Vec::new();

            enum State {
                None,
                Text,
                Thinking,
                ToolUse(i64), // tool_index
            }
            let mut state = State::None;
            let mut current_block_index = 0;
            let mut final_usage = Usage::default();

            // Send message_start
            yield StreamEvent::MessageStart {
                message: AnthropicResponse {
                    id: "msg_openai_placeholder".to_string(),
                    msg_type: "message".to_string(),
                    role: "assistant".to_string(),
                    content: vec![],
                    model: request.model.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                }
            };

            let mut pending_stop_reason = None;

            while let Some(item) = stream.next().await {
                let chunk = item?;
                debug!("chunk: {}", String::from_utf8_lossy(&chunk));
                buffer.extend_from_slice(&chunk);

                while let Some(pos) = buffer.windows(2).position(|w| w == b"\n\n") {
                    let line_bytes = buffer.drain(..pos).collect::<Vec<u8>>();
                    buffer.drain(..2); // remove \n\n

                    let line = String::from_utf8(line_bytes).unwrap_or_default();

                    if line.starts_with("data:") {
                        let data = line["data:".len()..].trim();
                        if data == "[DONE]" {
                            break;
                        }

                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            // Handle usage if present
                            if let Some(usage) = json.get("usage") {
                                let input_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                let output_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                final_usage.input_tokens = input_tokens;
                                final_usage.output_tokens = output_tokens;

                                if let Some(prompt_details) = usage.get("prompt_tokens_details") {
                                    final_usage.cache_read_input_tokens = prompt_details.get("cached_tokens").and_then(|v| v.as_u64()).map(|v| v as u32);
                                    final_usage.cache_creation_input_tokens = prompt_details.get("cache_write_tokens").and_then(|v| v.as_u64()).map(|v| v as u32);
                                    // OpenAI doesn't map exactly to cache_creation, often 0 or separate logic needed.
                                    // For now, map what we can finding in prompt_tokens_details.
                                    // Anthropic uses `cache_creation_input_tokens`. OpenAI usually has `cached_tokens`.
                                    // There is no direct "cache_creation" equivalent in standard OpenAI response yet commonly exposed.
                                    // We will leave creation as None or 0 if not found.
                                }
                            }

                            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                                if let Some(choice) = choices.first() {
                                    if let Some(delta) = choice.get("delta") {
                                        // 1. Handle Thinking (DeepSeek style)
                                        if let Some(reasoning) = delta.get("reasoning").and_then(|s| s.as_str()) {
                                            if !matches!(state, State::Thinking) {
                                                if !matches!(state, State::None) {
                                                    yield StreamEvent::ContentBlockStop { index: current_block_index };
                                                    current_block_index += 1;
                                                }
                                                state = State::Thinking;
                                                yield StreamEvent::ContentBlockStart {
                                                    index: current_block_index,
                                                    content_block: ContentBlock::Thinking { thinking: "".to_string(), signature: Some("".to_string()) },
                                                };
                                            }
                                            yield StreamEvent::ContentBlockDelta {
                                                index: current_block_index,
                                                delta: ContentBlockDelta::ThinkingDelta { thinking: reasoning.to_string() },
                                            };
                                        }

                                        // Handle signature (DeepSeek style)
                                        if let Some(signature) = delta.get("signature").and_then(|s| s.as_str()) {
                                             if matches!(state, State::Thinking) {
                                                 yield StreamEvent::ContentBlockDelta {
                                                    index: current_block_index,
                                                    delta: ContentBlockDelta::SignatureDelta { signature: signature.to_string() },
                                                };
                                             }
                                        }

                                        // 2. Handle Text
                                        if let Some(content) = delta.get("content").and_then(|s| s.as_str()) {
                                            if !content.is_empty() {
                                                if !matches!(state, State::Text) {
                                                    if !matches!(state, State::None) {
                                                        yield StreamEvent::ContentBlockStop { index: current_block_index };
                                                        current_block_index += 1;
                                                    }
                                                    state = State::Text;
                                                    yield StreamEvent::ContentBlockStart {
                                                        index: current_block_index,
                                                        content_block: ContentBlock::Text { text: "".to_string(), cache_control: None },
                                                    };
                                                }
                                                yield StreamEvent::ContentBlockDelta {
                                                    index: current_block_index,
                                                    delta: ContentBlockDelta::TextDelta { text: content.to_string() },
                                                };
                                            }
                                        }

                                        // 3. Handle Tool Calls
                                        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                                            for tc in tool_calls {
                                                let index = tc.get("index").and_then(|i| i.as_i64()).unwrap_or(0);

                                                let is_new_tool = if let State::ToolUse(current_idx) = state {
                                                    current_idx != index
                                                } else {
                                                    true
                                                };

                                                if is_new_tool {
                                                     if !matches!(state, State::None) {
                                                        yield StreamEvent::ContentBlockStop { index: current_block_index };
                                                        current_block_index += 1;
                                                    }
                                                    state = State::ToolUse(index);

                                                    let id = tc.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                    let name = tc.get("function").and_then(|f| f.get("name")).and_then(|s| s.as_str()).unwrap_or("").to_string();

                                                    // Only emit start if we have ID (start of tool call)
                                                    if !id.is_empty() {
                                                        yield StreamEvent::ContentBlockStart {
                                                            index: current_block_index,
                                                            content_block: ContentBlock::ToolUse {
                                                                id,
                                                                name,
                                                                input: json!({}),
                                                                thought_signature: None,
                                                                cache_control: None,
                                                            },
                                                        };
                                                    }
                                                }

                                                if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|s| s.as_str()) {
                                                     yield StreamEvent::ContentBlockDelta {
                                                        index: current_block_index,
                                                        delta: ContentBlockDelta::InputJsonDelta { partial_json: args.to_string() },
                                                    };
                                                }
                                            }
                                        }
                                    }

                                    // Handle finish_reason
                                    if let Some(finish_reason) = choice.get("finish_reason").and_then(|s| s.as_str()) {
                                        if !matches!(state, State::None) {
                                            yield StreamEvent::ContentBlockStop { index: current_block_index };
                                            state = State::None;
                                        }

                                        pending_stop_reason = Some(match finish_reason {
                                            "stop" => "end_turn",
                                            "length" => "max_tokens",
                                            "tool_calls" => "tool_use",
                                            _ => "end_turn",
                                        }.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some(stop_reason) = pending_stop_reason {
                let mut input_tokens = final_usage.input_tokens;
                let cache_read = final_usage.cache_read_input_tokens.unwrap_or(0);
                let cache_creation = final_usage.cache_creation_input_tokens.unwrap_or(0);
                if input_tokens >= cache_read + cache_creation {
                    input_tokens -= cache_read + cache_creation;
                }
                yield StreamEvent::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: Some(stop_reason),
                        stop_sequence: None,
                    },
                    usage: MessageDeltaUsage {
                        output_tokens: final_usage.output_tokens,
                        input_tokens: Some(input_tokens),
                        cache_creation_input_tokens: final_usage.cache_creation_input_tokens,
                        cache_read_input_tokens: final_usage.cache_read_input_tokens,
                    },
                };
                yield StreamEvent::MessageStop;
            }

        };

        Ok(Box::pin(output_stream))
    }

    async fn count_tokens(
        &self,
        request: AnthropicRequest,
    ) -> Result<u32, Box<dyn Error + Send + Sync>> {
        // Same approximation as Bedrock for now
        let text_len: usize = request
            .messages
            .iter()
            .map(|m| match &m.content {
                MessageContent::String(s) => s.len(),
                MessageContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text, .. } => text.len(),
                        _ => 0,
                    })
                    .sum(),
            })
            .sum();

        Ok((text_len / 4) as u32)
    }
}
