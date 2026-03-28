use crate::adapter::ModelAdapter;
use crate::handlers::HandlerError;
use crate::models::{ModelInfo, Provider, Transport};
use crate::services::ModelRegistry;
use axum::http::StatusCode;
use std::sync::Arc;

#[derive(Clone)]
pub struct RequestContext {
    pub resolved_model: String,
    pub adapter: Arc<dyn ModelAdapter + Send + Sync>,
    pub transport_model_id: String,
    pub provider: Provider,
    pub model_info: ModelInfo,
    pub transport: Transport,
}

impl RequestContext {
    pub fn uses_openai_responses_transport(&self) -> bool {
        self.transport.protocol == crate::models::TransportProtocol::OpenAIResponses
    }

    pub fn uses_bedrock_transport(&self) -> bool {
        self.transport.protocol == crate::models::TransportProtocol::Bedrock
    }

    pub fn ensure_endpoint_supported(&self, endpoint: &'static str) -> Result<(), HandlerError> {
        let _ = endpoint;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatCompletionsRoute {
    AnthropicAdapter,
    OpenAiPassthrough,
    GeminiDirectAdapter,
    OpenAIResponsesAdapter,
}

impl ChatCompletionsRoute {
    pub fn resolve(ctx: &RequestContext) -> Self {
        if ctx.transport.protocol == crate::models::TransportProtocol::Gemini {
            Self::GeminiDirectAdapter
        } else if ctx.transport.protocol == crate::models::TransportProtocol::OpenAIResponses {
            Self::OpenAIResponsesAdapter
        } else if ctx.transport.protocol == crate::models::TransportProtocol::Anthropic
            || ctx.transport.protocol == crate::models::TransportProtocol::Bedrock
            || ctx.provider == Provider::LiteLLM
        {
            Self::AnthropicAdapter
        } else {
            Self::OpenAiPassthrough
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponsesRoute {
    OpenAiResponsesPassthrough,
    AnthropicAdapter,
    GeminiDirectAdapter,
    OpenAIChatAdapter,
}

impl ResponsesRoute {
    pub fn resolve(ctx: &RequestContext) -> Self {
        if ctx.provider == Provider::OpenAI
            && ctx.transport.protocol == crate::models::TransportProtocol::OpenAIResponses
        {
            Self::OpenAiResponsesPassthrough
        } else if ctx.transport.protocol == crate::models::TransportProtocol::Gemini {
            Self::GeminiDirectAdapter
        } else if ctx.transport.protocol == crate::models::TransportProtocol::OpenAI {
            Self::OpenAIChatAdapter
        } else {
            Self::AnthropicAdapter
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeminiHandlerRoute {
    AnthropicAdapter,
    GeminiPassthrough,
    OpenAIChatAdapter,
}

impl GeminiHandlerRoute {
    pub fn resolve(ctx: &RequestContext) -> Self {
        match ctx.transport.protocol {
            crate::models::TransportProtocol::Gemini => Self::GeminiPassthrough,
            crate::models::TransportProtocol::OpenAI
            | crate::models::TransportProtocol::OpenAIResponses => Self::OpenAIChatAdapter,
            _ => Self::AnthropicAdapter,
        }
    }
}

impl ModelRegistry {
    fn build_request_context(&self, resolved_model: &str) -> Result<RequestContext, HandlerError> {
        let model_info = self.pricing.get(resolved_model).ok_or_else(|| {
            HandlerError::typed(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Model '{}' not found", resolved_model),
            )
        })?;

        let adapter = self
            .adapters
            .get(resolved_model)
            .ok_or_else(|| {
                HandlerError::typed(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    format!("Model '{}' not found in adapters", resolved_model),
                )
            })?
            .clone();

        let transport = model_info.transport.as_ref().ok_or_else(|| {
            HandlerError::typed(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Transport not found",
            )
        })?;

        Ok(RequestContext {
            resolved_model: resolved_model.to_string(),
            adapter,
            transport_model_id: transport.model_id.clone(),
            provider: transport.provider,
            model_info: model_info.clone(),
            transport: transport.clone(),
        })
    }

    pub fn resolve_context(&self, model_name: &str) -> Result<RequestContext, HandlerError> {
        self.build_request_context(model_name)
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatCompletionsRoute, GeminiHandlerRoute, RequestContext};
    use crate::adapter::ModelAdapter;
    use crate::models::{ModelInfo, Provider, Transport, TransportProtocol};
    use crate::pricing::Currency;
    use crate::types::{AnthropicRequest, AnthropicResponse, StreamEvent};
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use rust_decimal::Decimal;
    use std::error::Error;
    use std::sync::Arc;

    struct TestAdapter;

    #[async_trait]
    impl ModelAdapter for TestAdapter {
        async fn send_message(
            &self,
            _request: AnthropicRequest,
        ) -> Result<AnthropicResponse, Box<dyn Error + Send + Sync>> {
            unreachable!("route tests should not call adapter methods")
        }

        async fn send_message_stream(
            &self,
            _request: AnthropicRequest,
            _provider: Provider,
        ) -> Result<
            BoxStream<'static, Result<StreamEvent, Box<dyn Error + Send + Sync>>>,
            Box<dyn Error + Send + Sync>,
        > {
            Ok(Box::pin(stream::empty()))
        }

        async fn count_tokens(
            &self,
            _request: AnthropicRequest,
        ) -> Result<u32, Box<dyn Error + Send + Sync>> {
            unreachable!("route tests should not call adapter methods")
        }
    }

    fn build_request_context(protocol: TransportProtocol) -> RequestContext {
        RequestContext {
            resolved_model: "test-model".to_string(),
            adapter: Arc::new(TestAdapter),
            transport_model_id: "test-model".to_string(),
            provider: Provider::OpenAI,
            model_info: ModelInfo {
                transport: Some(Transport {
                    provider: Provider::OpenAI,
                    model_id: "test-model".to_string(),
                    protocol,
                    base_url: Some("https://example.com/v1".to_string()),
                    api_key: Some("test-key".to_string()),
                }),
                currency: Currency::USD,
                input_pricing_per_mtoken: Decimal::ZERO,
                cache_write_pricing_per_mtoken: Decimal::ZERO,
                cache_read_pricing_per_mtoken: Decimal::ZERO,
                reasoning_pricing_per_mtoken: Decimal::ZERO,
                output_pricing_per_mtoken: Decimal::ZERO,
                above: 0,
                input_pricing_per_mtoken_above: Decimal::ZERO,
                cache_write_pricing_per_mtoken_above: Decimal::ZERO,
                cache_read_pricing_per_mtoken_above: Decimal::ZERO,
                reasoning_pricing_per_mtoken_above: Decimal::ZERO,
                output_pricing_per_mtoken_above: Decimal::ZERO,
            },
            transport: Transport {
                provider: Provider::OpenAI,
                model_id: "test-model".to_string(),
                protocol,
                base_url: Some("https://example.com/v1".to_string()),
                api_key: Some("test-key".to_string()),
            },
        }
    }

    #[test]
    fn openai_responses_transport_allows_messages_endpoint() {
        let ctx = build_request_context(TransportProtocol::OpenAIResponses);
        ctx.ensure_endpoint_supported("/v1/messages")
            .expect("endpoint should be supported");
    }

    #[test]
    fn openai_responses_transport_allows_chat_completions_endpoint() {
        let ctx = build_request_context(TransportProtocol::OpenAIResponses);
        ctx.ensure_endpoint_supported("/v1/chat/completions")
            .expect("endpoint should be supported");
    }

    #[test]
    fn openai_responses_transport_uses_openai_responses_adapter_route_for_chat_completions() {
        let ctx = build_request_context(TransportProtocol::OpenAIResponses);
        assert_eq!(
            ChatCompletionsRoute::resolve(&ctx),
            ChatCompletionsRoute::OpenAIResponsesAdapter
        );
    }

    #[test]
    fn openai_responses_transport_allows_responses_endpoint() {
        let ctx = build_request_context(TransportProtocol::OpenAIResponses);
        ctx.ensure_endpoint_supported("/v1/responses")
            .expect("endpoint should be supported");
    }

    #[test]
    fn bedrock_transport_allows_chat_completions_endpoint() {
        let ctx = build_request_context(TransportProtocol::Bedrock);
        ctx.ensure_endpoint_supported("/v1/chat/completions")
            .expect("endpoint should be supported");
    }

    #[test]
    fn bedrock_transport_allows_responses_endpoint() {
        let ctx = build_request_context(TransportProtocol::Bedrock);
        ctx.ensure_endpoint_supported("/v1/responses")
            .expect("endpoint should be supported");
    }

    #[test]
    fn bedrock_transport_allows_messages_endpoint() {
        let ctx = build_request_context(TransportProtocol::Bedrock);
        ctx.ensure_endpoint_supported("/v1/messages")
            .expect("endpoint should be supported");
    }

    #[test]
    fn google_transport_uses_gemini_direct_adapter_route_for_chat_completions() {
        let ctx = build_request_context(TransportProtocol::Gemini);
        assert_eq!(
            ChatCompletionsRoute::resolve(&ctx),
            ChatCompletionsRoute::GeminiDirectAdapter
        );
    }

    #[test]
    fn openai_responses_transport_uses_openai_chat_adapter_route_for_gemini_endpoints() {
        let ctx = build_request_context(TransportProtocol::OpenAIResponses);
        assert_eq!(
            GeminiHandlerRoute::resolve(&ctx),
            GeminiHandlerRoute::OpenAIChatAdapter
        );
    }

    #[test]
    fn openai_transport_uses_openai_chat_adapter_route_for_gemini_endpoints() {
        let ctx = build_request_context(TransportProtocol::OpenAI);
        assert_eq!(
            GeminiHandlerRoute::resolve(&ctx),
            GeminiHandlerRoute::OpenAIChatAdapter
        );
    }
}
