mod chat;
mod gemini_openai_stream;
mod gemini_responses_stream;
mod messages;
mod passthrough_stream;
mod responses_openai_chat_stream;
mod responses_stream;

use crate::models::Provider;
use crate::types::{StreamEvent, Usage};

pub(crate) use self::chat::{handle_anthropic_adapter_route, handle_openai_passthrough_route};
pub(crate) use self::gemini_openai_stream::{
    gemini_sse_to_openai_chat_sse, openai_chat_sse_to_gemini_sse,
};
pub(crate) use self::gemini_responses_stream::gemini_stream_to_responses_sse;
pub(crate) use self::messages::{handle_messages_non_streaming, handle_messages_streaming};
pub(crate) use self::passthrough_stream::forward_sse_with_usage;
pub(crate) use self::responses_openai_chat_stream::{
    openai_chat_sse_to_responses_sse, responses_sse_to_openai_chat_sse,
};
pub(crate) use self::responses_stream::anthropic_stream_to_responses_sse;

pub(crate) fn merge_stream_usage(
    accumulated_usage: &mut Usage,
    event: &StreamEvent,
    provider: Provider,
) {
    match event {
        StreamEvent::MessageStart { message } => {
            if provider == crate::models::Provider::LiteLLM {
                accumulated_usage.input_tokens = message.usage.input_tokens;
                accumulated_usage.cache_creation_input_tokens =
                    message.usage.cache_creation_input_tokens;
                accumulated_usage.cache_read_input_tokens = message.usage.cache_read_input_tokens;
            }
        }
        StreamEvent::MessageDelta { usage, .. } => {
            if provider != crate::models::Provider::LiteLLM {
                accumulated_usage.input_tokens += usage.input_tokens.unwrap_or(0);
                accumulated_usage.cache_read_input_tokens = Some(
                    accumulated_usage.cache_read_input_tokens.unwrap_or(0)
                        + usage.cache_read_input_tokens.unwrap_or(0),
                );
                accumulated_usage.cache_creation_input_tokens = Some(
                    accumulated_usage.cache_creation_input_tokens.unwrap_or(0)
                        + usage.cache_creation_input_tokens.unwrap_or(0),
                );
            }
            accumulated_usage.output_tokens += usage.output_tokens;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::merge_stream_usage;
    use crate::models::Provider;
    use crate::types::{
        AnthropicResponse, ContentBlock, MessageDelta, MessageDeltaUsage, StreamEvent, Usage,
    };

    #[test]
    fn merge_stream_usage_keeps_message_start_usage_for_litellm() {
        let mut usage = Usage::default();

        merge_stream_usage(
            &mut usage,
            &StreamEvent::MessageStart {
                message: AnthropicResponse {
                    id: "msg_1".to_string(),
                    msg_type: "message".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ContentBlock::Text {
                        text: String::new(),
                        cache_control: None,
                    }],
                    model: "test-model".to_string(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage {
                        input_tokens: 120,
                        cache_creation_input_tokens: Some(30),
                        cache_read_input_tokens: Some(20),
                        output_tokens: 0,
                    },
                },
            },
            Provider::LiteLLM,
        );

        assert_eq!(usage.input_tokens, 120);
        assert_eq!(usage.cache_creation_input_tokens, Some(30));
        assert_eq!(usage.cache_read_input_tokens, Some(20));
    }

    #[test]
    fn merge_stream_usage_accumulates_message_delta_usage_for_non_litellm() {
        let mut usage = Usage {
            input_tokens: 120,
            cache_creation_input_tokens: Some(30),
            cache_read_input_tokens: Some(20),
            output_tokens: 5,
        };

        merge_stream_usage(
            &mut usage,
            &StreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason: Some("end_turn".to_string()),
                    stop_sequence: None,
                },
                usage: MessageDeltaUsage {
                    output_tokens: 45,
                    input_tokens: Some(70),
                    cache_creation_input_tokens: Some(10),
                    cache_read_input_tokens: Some(5),
                },
            },
            Provider::OpenAI,
        );

        assert_eq!(usage.input_tokens, 190);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, Some(40));
        assert_eq!(usage.cache_read_input_tokens, Some(25));
    }

    #[test]
    fn merge_stream_usage_uses_latest_message_delta_usage_for_litellm() {
        let mut usage = Usage {
            input_tokens: 120,
            cache_creation_input_tokens: Some(30),
            cache_read_input_tokens: Some(20),
            output_tokens: 5,
        };

        merge_stream_usage(
            &mut usage,
            &StreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason: Some("end_turn".to_string()),
                    stop_sequence: None,
                },
                usage: MessageDeltaUsage {
                    output_tokens: 45,
                    input_tokens: Some(70),
                    cache_creation_input_tokens: Some(10),
                    cache_read_input_tokens: Some(5),
                },
            },
            Provider::LiteLLM,
        );

        assert_eq!(usage.input_tokens, 70);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
        assert_eq!(usage.cache_read_input_tokens, Some(5));
    }
}
