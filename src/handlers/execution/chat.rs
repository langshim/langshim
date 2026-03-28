use crate::handlers::execution::{forward_sse_with_usage, merge_stream_usage};
use crate::handlers::translation::convert_stream_event_to_openai;
use crate::services::{AppState, RequestContext, UsageBreakdown, spawn_usage_recording};
use crate::types::StreamEvent;
use async_openai::types::chat::CreateChatCompletionRequest;
use axum::{
    extract::Json,
    http::StatusCode,
    response::{IntoResponse, Response, Sse, sse::Event},
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

fn prepare_openai_passthrough_request(
    mut request: CreateChatCompletionRequest,
    ctx: &RequestContext,
) -> CreateChatCompletionRequest {
    request.model = ctx.transport_model_id.clone();

    if request.max_tokens.is_some() && request.max_completion_tokens.is_none() {
        request.max_completion_tokens = request.max_tokens;
        request.max_tokens = None;
    }

    // Remove reasoning_effort — not supported on standard chat/completions endpoints
    request.reasoning_effort = None;

    // Remove stream_options when not streaming — only meaningful with stream: true
    if !request.stream.unwrap_or(false) {
        request.stream_options = None;
    }

    request
}

fn sanitize_openai_passthrough_request_value(
    request: &CreateChatCompletionRequest,
) -> serde_json::Value {
    let mut request_value =
        serde_json::to_value(request).expect("chat completion request should serialize");

    if let Some(map) = request_value.as_object_mut() {
        map.remove("reasoning_effort");
        if !request.stream.unwrap_or(false) {
            map.remove("stream_options");
        }
    }

    request_value
}

pub(crate) async fn handle_anthropic_adapter_route(
    state: Arc<AppState>,
    ctx: RequestContext,
    request: CreateChatCompletionRequest,
    start_time: Instant,
) -> Result<Response, Response> {
    let anthropic_request = state
        .openai_adapter()
        .convert_to_anthropic(request.clone())
        .map_err(|e| {
            tracing::error!("Failed to convert request: {:?}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        })?;

    let mut adapter_request = anthropic_request;
    adapter_request.model = ctx.transport_model_id.clone();

    if request.stream.unwrap_or(false) {
        let stream_result = ctx
            .adapter
            .send_message_stream(adapter_request, ctx.provider)
            .await;

        match stream_result {
            Ok(stream) => {
                let sse_stream = async_stream::stream! {
                    use futures::StreamExt;
                    let mut pinned_stream = stream;
                    let mut accumulated_usage = crate::types::Usage::default();
                    let mut ttft: Option<i32> = None;
                    let mut stream_model: Option<String> = None;
                    while let Some(result) = pinned_stream.next().await {
                        match result {
                            Ok(event) => {
                                if ttft.is_none() {
                                    ttft = Some(start_time.elapsed().as_millis() as i32);
                                }
                                if let StreamEvent::MessageStart { message } = &event {
                                    stream_model = Some(message.model.clone());
                                }
                                merge_stream_usage(&mut accumulated_usage, &event, ctx.provider);
                                debug!("Received message: {:?}", event);
                                let model = stream_model
                                    .clone()
                                    .unwrap_or_else(|| ctx.resolved_model.clone());
                                let openai_data = convert_stream_event_to_openai(event, model);
                                if let Some(data) = openai_data {
                                    yield Ok::<_, axum::Error>(Event::default().json_data(data).unwrap());
                                }
                            }
                            Err(e) => {
                                tracing::error!("Stream error: {:?}", e);
                                yield Err(axum::Error::new(e));
                            }
                        }
                    }

                    spawn_usage_recording(
                        state.clone(),
                        ctx.clone(),
                        UsageBreakdown::from_anthropic_usage(&accumulated_usage),
                        ttft,
                        Some(start_time.elapsed().as_millis() as i32),
                    );

                    yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));
                };

                Ok(Sse::new(sse_stream)
                    .keep_alive(axum::response::sse::KeepAlive::default())
                    .into_response())
            }
            Err(e) => {
                tracing::error!("Service error: {:?}", e);
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response())
            }
        }
    } else {
        match ctx.adapter.send_message(adapter_request).await {
            Ok(response) => {
                state
                    .billing_service()
                    .record_usage_and_charge(
                        &ctx,
                        UsageBreakdown::from_anthropic_usage(&response.usage),
                        None,
                        Some(start_time.elapsed().as_millis() as i32),
                    )
                    .await;

                let openai_response = state.openai_adapter().convert_response_to_openai(response);
                Ok(Json(openai_response).into_response())
            }
            Err(e) => {
                tracing::error!("Service error: {:?}", e);
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response())
            }
        }
    }
}

pub(crate) async fn handle_openai_passthrough_route(
    state: Arc<AppState>,
    ctx: RequestContext,
    request: CreateChatCompletionRequest,
    start_time: Instant,
) -> Result<Response, Response> {
    let url = format!(
        "{}/chat/completions",
        ctx.transport
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
    );
    let api_key = ctx.transport.api_key.as_deref().unwrap_or("");
    let modified_request = prepare_openai_passthrough_request(request, &ctx);
    let request_value = sanitize_openai_passthrough_request_value(&modified_request);

    if modified_request.stream.unwrap_or(false) {
        let client = state.http_client();
        let request_builder = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_value);

        let response = request_builder.send().await.map_err(|e| {
            tracing::error!("Request error: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        })?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("OpenAI API Error: {}", text)})),
            )
                .into_response());
        }

        Ok(forward_sse_with_usage(
            state.clone(),
            ctx.clone(),
            start_time,
            response.bytes_stream(),
            true,
            |sse| {
                if sse.get("usage").is_some() {
                    Some(UsageBreakdown::from_openai_chat_json(sse))
                } else {
                    None
                }
            },
        ))
    } else {
        let client = state.http_client();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_value)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("Request error: {:?}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )
                    .into_response()
            })?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("OpenAI API Error: {}", text)})),
            )
                .into_response());
        }

        let bytes = response.bytes().await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        })?;

        let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        })?;

        state
            .billing_service()
            .record_usage_and_charge(
                &ctx,
                UsageBreakdown::from_openai_chat_json(&json),
                None,
                Some(start_time.elapsed().as_millis() as i32),
            )
            .await;

        Ok(Json(json).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::{prepare_openai_passthrough_request, sanitize_openai_passthrough_request_value};
    use crate::models::{ModelInfo, Provider, Transport, TransportProtocol};
    use crate::pricing::Currency;
    use crate::services::RequestContext;
    use async_openai::types::chat::{ChatCompletionStreamOptions, CreateChatCompletionRequest};
    use rust_decimal::Decimal;
    use serde_json::json;
    use std::sync::Arc;

    fn build_request_context() -> RequestContext {
        RequestContext {
            resolved_model: "gpt-4.1-mini".to_string(),
            adapter: Arc::new(crate::adapter::openai::OpenAIAdapter::new(
                "test-key".to_string(),
                Some("https://example.com/v1".to_string()),
            )),
            transport_model_id: "gpt-4.1-mini".to_string(),
            provider: Provider::OpenAI,
            model_info: ModelInfo {
                transport: Some(Transport {
                    provider: Provider::OpenAI,
                    model_id: "gpt-4.1-mini".to_string(),
                    protocol: TransportProtocol::OpenAI,
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
                model_id: "gpt-4.1-mini".to_string(),
                protocol: TransportProtocol::OpenAI,
                base_url: Some("https://example.com/v1".to_string()),
                api_key: Some("test-key".to_string()),
            },
        }
    }

    #[test]
    fn prepare_openai_passthrough_request_rewrites_model_and_normalizes_max_tokens() {
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "standard",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 256
        }))
        .expect("request should deserialize");

        let prepared = prepare_openai_passthrough_request(request, &build_request_context());

        assert_eq!(prepared.model, "gpt-4.1-mini");
        assert_eq!(prepared.max_completion_tokens, Some(256));
        assert_eq!(prepared.max_tokens, None);
        assert_eq!(prepared.stream, Some(true));
    }

    #[test]
    fn prepare_openai_passthrough_request_preserves_existing_max_completion_tokens() {
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "standard",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "max_completion_tokens": 512
        }))
        .expect("request should deserialize");

        let prepared = prepare_openai_passthrough_request(request, &build_request_context());

        assert_eq!(prepared.model, "gpt-4.1-mini");
        assert_eq!(prepared.max_completion_tokens, Some(512));
        assert_eq!(prepared.max_tokens, Some(128));
    }

    #[test]
    fn sanitize_openai_passthrough_request_value_removes_reasoning_effort() {
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "standard",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "medium"
        }))
        .expect("request should deserialize");

        let value = sanitize_openai_passthrough_request_value(&request);

        assert!(value.get("reasoning_effort").is_none());
    }

    #[test]
    fn sanitize_openai_passthrough_request_value_removes_stream_options_when_not_streaming() {
        let mut request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "standard",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .expect("request should deserialize");
        request.stream_options = Some(ChatCompletionStreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        });

        let value = sanitize_openai_passthrough_request_value(&request);

        assert!(value.get("stream_options").is_none());
    }
}
