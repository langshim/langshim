use super::error::parse_json;
use super::execution::{
    gemini_sse_to_openai_chat_sse, handle_anthropic_adapter_route, handle_openai_passthrough_route,
    responses_sse_to_openai_chat_sse,
};
use super::translation::{
    convert_chat_request_to_responses, convert_gemini_response_to_openai_chat,
    convert_openai_chat_to_gemini_request, convert_responses_response_to_openai_chat,
};
use crate::auth::Authenticated;
use crate::services::{AppState, ChatCompletionsRoute, UsageBreakdown};
use crate::types::GeminiGenerateContentResponse;
use async_openai::types::chat::CreateChatCompletionRequest;
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    _auth: Authenticated,
    body: Bytes,
) -> Result<impl IntoResponse, Response> {
    let request: CreateChatCompletionRequest = parse_json(&body).map_err(|e| e.into_response())?;
    let start_time = Instant::now();
    let model_name = request.model.clone();

    let ctx = state
        .resolve_context(&model_name)
        .map_err(|e| e.into_response())?;
    ctx.ensure_endpoint_supported("/v1/chat/completions")
        .map_err(|e| e.into_response())?;

    state
        .billing_service()
        .ensure_balance()
        .await
        .map_err(|e| e.into_response())?;

    match ChatCompletionsRoute::resolve(&ctx) {
        ChatCompletionsRoute::AnthropicAdapter => {
            handle_anthropic_adapter_route(state, ctx, request, start_time).await
        }
        ChatCompletionsRoute::OpenAiPassthrough => {
            handle_openai_passthrough_route(state, ctx, request, start_time).await
        }
        ChatCompletionsRoute::OpenAIResponsesAdapter => {
            handle_openai_responses_adapter_route(state, ctx, request, start_time).await
        }
        ChatCompletionsRoute::GeminiDirectAdapter => {
            let base_url = ctx
                .transport
                .base_url
                .as_deref()
                .unwrap_or("https://generativelanguage.googleapis.com")
                .trim_end_matches('/');
            let model_id = &ctx.transport_model_id;
            let normalized_model = if model_id.starts_with("models/") {
                model_id.clone()
            } else {
                format!("models/{}", model_id)
            };
            let api_key = ctx.transport.api_key.as_deref().unwrap_or("").to_string();

            let gemini_req = convert_openai_chat_to_gemini_request(&request);

            if request.stream.unwrap_or(false) {
                let url = format!(
                    "{}/v1beta/{}:streamGenerateContent?alt=sse",
                    base_url, normalized_model
                );
                let response = state
                    .http_client()
                    .post(&url)
                    .header("x-goog-api-key", &api_key)
                    .header("content-type", "application/json")
                    .json(&gemini_req)
                    .send()
                    .await
                    .map_err(|e| {
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
                        Json(json!({"error": format!("Gemini API Error: {}", text)})),
                    )
                        .into_response());
                }

                return Ok(
                    gemini_sse_to_openai_chat_sse(state, ctx, start_time, response).into_response(),
                );
            }

            let url = format!("{}/v1beta/{}:generateContent", base_url, normalized_model);
            let response = state
                .http_client()
                .post(&url)
                .header("x-goog-api-key", &api_key)
                .header("content-type", "application/json")
                .json(&gemini_req)
                .send()
                .await
                .map_err(|e| {
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
                    Json(json!({"error": format!("Gemini API Error: {}", text)})),
                )
                    .into_response());
            }

            let gemini_resp: GeminiGenerateContentResponse =
                response.json().await.map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": e.to_string()})),
                    )
                        .into_response()
                })?;

            let usage = UsageBreakdown {
                input_tokens: gemini_resp
                    .usage_metadata
                    .as_ref()
                    .and_then(|m| m.prompt_token_count)
                    .unwrap_or(0) as i32,
                output_tokens: gemini_resp
                    .usage_metadata
                    .as_ref()
                    .and_then(|m| m.candidates_token_count)
                    .unwrap_or(0) as i32,
                cache_read_tokens: gemini_resp
                    .usage_metadata
                    .as_ref()
                    .and_then(|m| m.cached_content_token_count)
                    .unwrap_or(0) as i32,
                cache_creation_tokens: 0,
                reasoning_tokens: 0,
            };

            state
                .billing_service()
                .record_usage_and_charge(
                    &ctx,
                    usage,
                    None,
                    Some(start_time.elapsed().as_millis() as i32),
                )
                .await;

            let openai_resp = convert_gemini_response_to_openai_chat(&model_name, gemini_resp);
            Ok(Json(openai_resp).into_response())
        }
    }
}

async fn handle_openai_responses_adapter_route(
    state: Arc<AppState>,
    ctx: crate::services::RequestContext,
    request: CreateChatCompletionRequest,
    start_time: Instant,
) -> Result<Response, Response> {
    let url = format!(
        "{}/responses",
        ctx.transport
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
            .trim_end_matches('/')
    );
    let api_key = ctx.transport.api_key.as_deref().unwrap_or("").to_string();
    let is_stream = request.stream.unwrap_or(false);

    let mut responses_req = convert_chat_request_to_responses(&request);
    responses_req.model = ctx.transport_model_id.clone();

    let client = state.http_client();
    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&responses_req)
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

    if is_stream {
        return Ok(responses_sse_to_openai_chat_sse(
            state.clone(),
            ctx.clone(),
            start_time,
            response,
        )
        .into_response());
    }

    let resp_json: serde_json::Value = response.json().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response()
    })?;

    let usage = UsageBreakdown {
        input_tokens: resp_json
            .get("usage")
            .and_then(|u| u.get("input_tokens"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        output_tokens: resp_json
            .get("usage")
            .and_then(|u| u.get("output_tokens"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
    };

    state
        .billing_service()
        .record_usage_and_charge(
            &ctx,
            usage,
            None,
            Some(start_time.elapsed().as_millis() as i32),
        )
        .await;

    let chat_resp = convert_responses_response_to_openai_chat(&resp_json);
    Ok(Json(chat_resp).into_response())
}
