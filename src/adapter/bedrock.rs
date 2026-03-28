use crate::adapter::ModelAdapter;
use crate::models::Provider;
use crate::types::{
    AnthropicRequest, AnthropicResponse, CacheControl, ContentBlock, ErrorDetails, Message,
    MessageContent, StreamEvent, SystemPrompt, Tool,
};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::config::Token;
use aws_sdk_bedrockruntime::error::DisplayErrorContext;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_smithy_runtime_api::client::auth::http::HTTP_BEARER_AUTH_SCHEME_ID;
use futures::stream::BoxStream;
use serde_json::Value;
use std::error::Error;

#[derive(Clone)]
pub struct BedrockAdapter {
    client: Client,
}

impl BedrockAdapter {
    pub async fn new(endpoint: Option<&str>, api_key: Option<&str>) -> Self {
        let mut config_loader = aws_config::defaults(BehaviorVersion::latest());

        if let Some(region) = infer_region_from_endpoint(endpoint) {
            config_loader = config_loader.region(aws_config::Region::new(region));
        }

        let sdk_config = config_loader.load().await;

        let mut client_config_builder = aws_sdk_bedrockruntime::config::Builder::from(&sdk_config);
        if let Some(url) = endpoint {
            client_config_builder = client_config_builder.endpoint_url(url);
        }
        if let Some(token) = api_key {
            client_config_builder = client_config_builder
                .token_provider(Token::new(token, None))
                .auth_scheme_preference([HTTP_BEARER_AUTH_SCHEME_ID]);
        }

        let client = Client::from_conf(client_config_builder.build());
        Self { client }
    }

    fn prepare_body(&self, request: &AnthropicRequest) -> Result<Vec<u8>, serde_json::Error> {
        let normalized_request = normalize_request_for_bedrock(request.clone());

        let mut body_json = serde_json::to_value(normalized_request)?;

        if let Some(obj) = body_json.as_object_mut() {
            obj.remove("model");
            obj.remove("stream");
            obj.remove("metadata");
            obj.insert(
                "anthropic_version".to_string(),
                Value::String("bedrock-2023-05-31".to_string()),
            );
        }

        serde_json::to_vec(&body_json)
    }
}

fn normalize_request_for_bedrock(mut request: AnthropicRequest) -> AnthropicRequest {
    request.messages = normalize_messages_for_bedrock(request.messages);
    ensure_last_message_cache_control(&mut request);
    request.system = normalize_system_prompt_for_bedrock(request.system);
    request.tools = normalize_tools_for_bedrock(request.tools);
    if request.temperature.is_some() && request.top_p.is_some() {
        // Bedrock Anthropic models reject requests that set both sampling controls.
        request.top_p = None;
    }
    let thinking_was_adaptive = request
        .thinking
        .as_ref()
        .is_some_and(|thinking| thinking.thinking_type == "adaptive");
    request.thinking = normalize_thinking_for_bedrock(request.thinking);
    if thinking_was_adaptive {
        request.max_tokens = request.max_tokens.saturating_add(32_000);
    }
    request
}

fn normalize_messages_for_bedrock(messages: Vec<Message>) -> Vec<Message> {
    messages
        .into_iter()
        .map(|mut message| {
            if let MessageContent::Blocks(blocks) = message.content {
                message.content =
                    MessageContent::Blocks(normalize_content_blocks_for_bedrock(blocks));
            }
            message
        })
        .collect()
}

fn normalize_content_blocks_for_bedrock(blocks: Vec<ContentBlock>) -> Vec<ContentBlock> {
    blocks
        .into_iter()
        .map(|block| match block {
            ContentBlock::ToolUse {
                id,
                name,
                input,
                cache_control,
                ..
            } => ContentBlock::ToolUse {
                id,
                name,
                input,
                thought_signature: None,
                cache_control,
            },
            other => other,
        })
        .collect()
}

fn ensure_last_message_cache_control(request: &mut AnthropicRequest) {
    let Some(last_message) = request.messages.last_mut() else {
        return;
    };

    match &mut last_message.content {
        MessageContent::String(text) => {
            let text = std::mem::take(text);
            last_message.content = MessageContent::Blocks(vec![crate::types::ContentBlock::Text {
                text,
                cache_control: Some(CacheControl {
                    control_type: "ephemeral".to_string(),
                    ttl: None,
                }),
            }]);
        }
        MessageContent::Blocks(blocks) => {
            let Some(last_block) = blocks.last_mut() else {
                return;
            };

            match last_block {
                crate::types::ContentBlock::Text { cache_control, .. }
                | crate::types::ContentBlock::Image { cache_control, .. }
                | crate::types::ContentBlock::ToolUse { cache_control, .. }
                | crate::types::ContentBlock::ToolResult { cache_control, .. } => {
                    if cache_control.is_none() {
                        *cache_control = Some(CacheControl {
                            control_type: "ephemeral".to_string(),
                            ttl: None,
                        });
                    }
                }
                crate::types::ContentBlock::Thinking { .. } => {}
            }
        }
    }
}

fn normalize_tools_for_bedrock(tools: Option<Vec<Tool>>) -> Option<Vec<Tool>> {
    tools.map(|tools| {
        tools
            .into_iter()
            .map(|mut tool| {
                tool.input_schema = normalize_tool_input_schema_for_bedrock(tool.input_schema);
                tool
            })
            .collect()
    })
}

fn normalize_tool_input_schema_for_bedrock(input_schema: Value) -> Value {
    match input_schema {
        Value::Object(mut obj) => {
            obj.entry("type".to_string())
                .or_insert_with(|| Value::String("object".to_string()));
            Value::Object(obj)
        }
        _ => serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }
}

fn normalize_system_prompt_for_bedrock(system: Option<SystemPrompt>) -> Option<SystemPrompt> {
    match system {
        Some(SystemPrompt::Blocks(blocks)) => {
            let text = blocks
                .into_iter()
                .filter_map(|block| match block {
                    crate::types::ContentBlock::Text { text, .. } => Some(text),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n");

            if text.is_empty() {
                None
            } else {
                Some(SystemPrompt::String(text))
            }
        }
        other => other,
    }
}

fn normalize_thinking_for_bedrock(
    thinking: Option<crate::types::Thinking>,
) -> Option<crate::types::Thinking> {
    match thinking {
        Some(mut thinking) if thinking.thinking_type == "adaptive" => {
            thinking.thinking_type = "enabled".to_string();
            thinking.budget_tokens = Some(32_000);
            Some(thinking)
        }
        other => other,
    }
}

fn infer_region_from_endpoint(endpoint: Option<&str>) -> Option<String> {
    let endpoint = endpoint?;
    let host = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(endpoint)
        .split('/')
        .next()
        .unwrap_or(endpoint)
        .split(':')
        .next()
        .unwrap_or(endpoint);

    host.split('.')
        .find(|segment| is_aws_region(segment))
        .map(ToString::to_string)
}

fn is_aws_region(value: &str) -> bool {
    let mut parts = value.split('-');
    let Some(part1) = parts.next() else {
        return false;
    };
    let Some(part2) = parts.next() else {
        return false;
    };
    let Some(part3) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }

    let valid_prefix = part1 == "us"
        || part1 == "eu"
        || part1 == "ap"
        || part1 == "sa"
        || part1 == "ca"
        || part1 == "me"
        || part1 == "af";

    valid_prefix && !part2.is_empty() && part3.chars().all(|char| char.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_last_message_cache_control, infer_region_from_endpoint,
        normalize_request_for_bedrock, normalize_system_prompt_for_bedrock,
        normalize_thinking_for_bedrock, normalize_tool_input_schema_for_bedrock,
    };
    use crate::types::{
        AnthropicRequest, CacheControl, ContentBlock, Message, MessageContent, SystemPrompt,
        Thinking,
    };
    use serde_json::Value;

    #[test]
    fn infers_region_from_standard_bedrock_runtime_endpoint() {
        assert_eq!(
            infer_region_from_endpoint(Some("https://bedrock-runtime.us-west-2.amazonaws.com")),
            Some("us-west-2".to_string())
        );
    }

    #[test]
    fn infers_region_from_endpoint_with_path() {
        assert_eq!(
            infer_region_from_endpoint(Some(
                "https://bedrock-runtime.ap-southeast-1.amazonaws.com/model/test"
            )),
            Some("ap-southeast-1".to_string())
        );
    }

    #[test]
    fn returns_none_for_non_aws_host() {
        assert_eq!(
            infer_region_from_endpoint(Some("https://bedrock.internal.example.com")),
            None
        );
    }

    #[test]
    fn flattens_system_blocks_into_string_for_bedrock() {
        let system = Some(SystemPrompt::Blocks(vec![
            ContentBlock::Text {
                text: "first".to_string(),
                cache_control: Some(CacheControl {
                    control_type: "ephemeral".to_string(),
                    ttl: None,
                }),
            },
            ContentBlock::Text {
                text: "second".to_string(),
                cache_control: None,
            },
        ]));

        let normalized = normalize_system_prompt_for_bedrock(system);

        assert!(matches!(
            normalized,
            Some(SystemPrompt::String(text)) if text == "first\n\nsecond"
        ));
    }

    #[test]
    fn rewrites_adaptive_thinking_for_bedrock() {
        let normalized = normalize_thinking_for_bedrock(Some(Thinking {
            thinking_type: "adaptive".to_string(),
            budget_tokens: None,
        }));

        assert!(matches!(
            normalized,
            Some(Thinking {
                thinking_type,
                budget_tokens: Some(32_000)
            }) if thinking_type == "enabled"
        ));
    }

    #[test]
    fn adaptive_thinking_adds_budget_to_max_tokens_in_prepared_body() {
        let request = AnthropicRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::String("hello".to_string()),
            }],
            max_tokens: 4096,
            metadata: None,
            stop_sequences: None,
            stream: Some(false),
            system: None,
            temperature: None,
            thinking: Some(Thinking {
                thinking_type: "adaptive".to_string(),
                budget_tokens: None,
            }),
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        let normalized = normalize_request_for_bedrock(request);

        assert_eq!(normalized.max_tokens, 36_096);
        assert!(matches!(
            normalized.thinking,
            Some(Thinking {
                thinking_type,
                budget_tokens: Some(32_000)
            }) if thinking_type == "enabled"
        ));
    }

    #[test]
    fn drops_top_p_when_temperature_is_also_set_for_bedrock() {
        let request = AnthropicRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::String("hello".to_string()),
            }],
            max_tokens: 1024,
            metadata: None,
            stop_sequences: None,
            stream: Some(true),
            system: None,
            temperature: Some(0.7),
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: Some(0.9),
            cache_control: None,
        };

        let normalized = normalize_request_for_bedrock(request);

        assert_eq!(normalized.temperature, Some(0.7));
        assert_eq!(normalized.top_p, None);
    }

    #[test]
    fn drops_tool_use_thought_signature_for_bedrock() {
        let request = AnthropicRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({ "query": "langshim" }),
                    thought_signature: Some("sig-123".to_string()),
                    cache_control: None,
                }]),
            }],
            max_tokens: 1024,
            metadata: None,
            stop_sequences: None,
            stream: Some(false),
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        let normalized = normalize_request_for_bedrock(request);

        let tool_use = normalized
            .messages
            .first()
            .and_then(|message| match &message.content {
                MessageContent::Blocks(blocks) => blocks.first(),
                MessageContent::String(_) => None,
            });

        assert!(matches!(
            tool_use,
            Some(ContentBlock::ToolUse {
                thought_signature: None,
                ..
            })
        ));
    }

    #[test]
    fn adds_cache_control_to_last_message_when_content_is_string() {
        let mut request = AnthropicRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::String("hello".to_string()),
            }],
            max_tokens: 1024,
            metadata: None,
            stop_sequences: None,
            stream: Some(false),
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        ensure_last_message_cache_control(&mut request);

        let cache_control = request
            .messages
            .first()
            .and_then(|message| match &message.content {
                MessageContent::Blocks(blocks) => blocks.first(),
                MessageContent::String(_) => None,
            });

        assert!(matches!(
            cache_control,
            Some(ContentBlock::Text {
                cache_control: Some(CacheControl { control_type, .. }),
                ..
            }) if control_type == "ephemeral"
        ));
    }

    #[test]
    fn preserves_existing_cache_control_on_last_message_block() {
        let mut request = AnthropicRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![ContentBlock::Text {
                    text: "hello".to_string(),
                    cache_control: Some(CacheControl {
                        control_type: "ephemeral".to_string(),
                        ttl: Some("1h".to_string()),
                    }),
                }]),
            }],
            max_tokens: 1024,
            metadata: None,
            stop_sequences: None,
            stream: Some(false),
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        ensure_last_message_cache_control(&mut request);

        let cache_control = request
            .messages
            .first()
            .and_then(|message| match &message.content {
                MessageContent::Blocks(blocks) => blocks.first(),
                MessageContent::String(_) => None,
            });

        assert!(matches!(
            cache_control,
            Some(ContentBlock::Text {
                cache_control: Some(CacheControl { ttl: Some(ttl), .. }),
                ..
            }) if ttl == "1h"
        ));
    }

    #[test]
    fn normalizes_empty_tool_schema_for_bedrock() {
        assert_eq!(
            normalize_tool_input_schema_for_bedrock(serde_json::json!({})),
            serde_json::json!({
                "type": "object"
            })
        );
    }

    #[test]
    fn adds_missing_type_to_tool_schema_for_bedrock() {
        assert_eq!(
            normalize_tool_input_schema_for_bedrock(serde_json::json!({
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
        );
    }

    #[test]
    fn replaces_non_object_tool_schema_for_bedrock() {
        assert_eq!(
            normalize_tool_input_schema_for_bedrock(Value::Null),
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        );
    }
}

#[async_trait]
impl ModelAdapter for BedrockAdapter {
    async fn send_message(
        &self,
        request: AnthropicRequest,
    ) -> Result<AnthropicResponse, Box<dyn Error + Send + Sync>> {
        let model_id = request.model.clone().replace("bedrock/", "");
        let body = self.prepare_body(&request)?;

        let response = self
            .client
            .invoke_model()
            .model_id(model_id)
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|error| std::io::Error::other(format!("{}", DisplayErrorContext(&error))))?;

        let response_body = response.body.into_inner();
        let anthropic_response: AnthropicResponse = serde_json::from_slice(&response_body)?;

        Ok(anthropic_response)
    }

    async fn send_message_stream(
        &self,
        request: AnthropicRequest,
        _provider: Provider,
    ) -> Result<
        BoxStream<'static, Result<StreamEvent, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    > {
        let model_id = request.model.clone().replace("bedrock/", "");
        let body = self.prepare_body(&request)?;

        let response = self
            .client
            .invoke_model_with_response_stream()
            .model_id(model_id)
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|error| std::io::Error::other(format!("{}", DisplayErrorContext(&error))))?;

        let mut stream = response.body;

        let output_stream = async_stream::try_stream! {
            loop {
                match stream.recv().await {
                    Ok(Some(aws_sdk_bedrockruntime::types::ResponseStream::Chunk(chunk))) => {
                        if let Some(bytes) = chunk.bytes {
                            let event_json: Value = serde_json::from_slice(bytes.as_ref())?;
                            if let Ok(stream_event) = serde_json::from_value::<StreamEvent>(event_json.clone()) {
                                yield stream_event;
                            }
                        }
                    }
                    Ok(Some(_)) => {} // Handle other event types
                    Ok(None) => break,
                    Err(e) => {
                        yield StreamEvent::Error {
                            error: ErrorDetails {
                                error_type: "bedrock_error".to_string(),
                                message: format!("{}", DisplayErrorContext(&e))
                            }
                        };
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(output_stream))
    }

    async fn count_tokens(
        &self,
        request: AnthropicRequest,
    ) -> Result<u32, Box<dyn Error + Send + Sync>> {
        // Approximation
        let text_len: usize = request
            .messages
            .iter()
            .map(|m| match &m.content {
                crate::types::MessageContent::String(s) => s.len(),
                crate::types::MessageContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| match b {
                        crate::types::ContentBlock::Text { text, .. } => text.len(),
                        _ => 0,
                    })
                    .sum(),
            })
            .sum();

        Ok((text_len / 4) as u32)
    }
}
