use super::error::parse_json;
use super::execution::{
    anthropic_stream_to_responses_sse, forward_sse_with_usage, gemini_stream_to_responses_sse,
    openai_chat_sse_to_responses_sse,
};
use super::translation::{
    convert_anthropic_to_responses, convert_gemini_response_to_responses,
    convert_openai_chat_to_responses, convert_responses_to_anthropic,
    convert_responses_to_gemini_request, convert_responses_to_openai_chat,
};
use crate::auth::Authenticated;
use crate::services::{AppState, ResponsesRoute, UsageBreakdown};
use crate::types::{GeminiGenerateContentResponse, ResponsesRequest};
use axum::{
    body::Bytes,
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::backtrace::Backtrace;
use std::sync::Arc;
use std::time::Instant;

pub async fn handle_responses(
    State(state): State<Arc<AppState>>,
    _auth: Authenticated,
    body: Bytes,
) -> Result<impl IntoResponse, Response> {
    let request: ResponsesRequest = parse_json(&body).map_err(|e| e.into_response())?;
    let start_time = Instant::now();
    let model_name = request.model.clone();

    let ctx = state
        .resolve_context(&model_name)
        .map_err(|e| e.into_response())?;
    ctx.ensure_endpoint_supported("/v1/responses")
        .map_err(|e| e.into_response())?;

    state
        .billing_service()
        .ensure_balance()
        .await
        .map_err(|e| e.into_response())?;

    match ResponsesRoute::resolve(&ctx) {
        ResponsesRoute::OpenAiResponsesPassthrough => {
            let url = format!(
                "{}/responses",
                ctx.transport
                    .base_url
                    .as_deref()
                    .unwrap_or("https://api.openai.com/v1")
                    .trim_end_matches('/')
            );
            let api_key = ctx.transport.api_key.as_deref().unwrap_or("");
            let client = state.http_client();
            let mut modified_request = request.clone();
            modified_request.model = ctx.transport_model_id.clone();

            if request.stream.unwrap_or(false) {
                let response = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .json(&modified_request)
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
                        Json(json!({"error": format!("OpenAI Responses API Error: {}", text)})),
                    )
                        .into_response());
                }

                return Ok(forward_sse_with_usage(
                    state.clone(),
                    ctx.clone(),
                    start_time,
                    response.bytes_stream(),
                    true,
                    |sse| Some(UsageBreakdown::from_responses_json(sse)),
                ));
            }

            let response = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&modified_request)
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
                    Json(json!({"error": format!("OpenAI Responses API Error: {}", text)})),
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
                    UsageBreakdown::from_responses_json(&json),
                    None,
                    Some(start_time.elapsed().as_millis() as i32),
                )
                .await;

            return Ok(Json(json).into_response());
        }
        ResponsesRoute::GeminiDirectAdapter => {
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
            let gemini_request = convert_responses_to_gemini_request(&request);
            let client = state.http_client();

            if request.stream.unwrap_or(false) {
                let url = format!(
                    "{}/v1beta/{}:streamGenerateContent?alt=sse",
                    base_url, normalized_model
                );
                let response = client
                    .post(&url)
                    .header("x-goog-api-key", &api_key)
                    .header("content-type", "application/json")
                    .json(&gemini_request)
                    .send()
                    .await
                    .map_err(|e| {
                        tracing::error!("Gemini stream request error: {:?}", e);
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

                return Ok(gemini_stream_to_responses_sse(
                    state.clone(),
                    ctx.clone(),
                    request.clone(),
                    start_time,
                    response,
                ));
            }

            let url = format!("{}/v1beta/{}:generateContent", base_url, normalized_model);
            let response = client
                .post(&url)
                .header("x-goog-api-key", &api_key)
                .header("content-type", "application/json")
                .json(&gemini_request)
                .send()
                .await
                .map_err(|e| {
                    tracing::error!("Gemini request error: {:?}", e);
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

            let gemini_response: GeminiGenerateContentResponse =
                response.json().await.map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": e.to_string()})),
                    )
                        .into_response()
                })?;

            let responses_response =
                convert_gemini_response_to_responses(&model_name, gemini_response);

            let usage_breakdown = UsageBreakdown {
                input_tokens: responses_response
                    .usage
                    .as_ref()
                    .and_then(|u| u.input_tokens)
                    .unwrap_or(0) as i32,
                output_tokens: responses_response
                    .usage
                    .as_ref()
                    .and_then(|u| u.output_tokens)
                    .unwrap_or(0) as i32,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                reasoning_tokens: 0,
            };

            state
                .billing_service()
                .record_usage_and_charge(
                    &ctx,
                    usage_breakdown,
                    None,
                    Some(start_time.elapsed().as_millis() as i32),
                )
                .await;

            return Ok(Json(responses_response).into_response());
        }
        ResponsesRoute::OpenAIChatAdapter => {
            let url = format!(
                "{}/chat/completions",
                ctx.transport
                    .base_url
                    .as_deref()
                    .unwrap_or("https://api.openai.com/v1")
                    .trim_end_matches('/')
            );
            let api_key = ctx.transport.api_key.as_deref().unwrap_or("").to_string();
            let mut chat_req = convert_responses_to_openai_chat(&request);
            chat_req.model = ctx.transport_model_id.clone();

            if request.stream.unwrap_or(false) {
                let response = state
                    .http_client()
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .json(&chat_req)
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
                        Json(json!({"error": format!("OpenAI Chat API Error: {}", text)})),
                    )
                        .into_response());
                }

                return Ok(openai_chat_sse_to_responses_sse(
                    state.clone(),
                    ctx.clone(),
                    request.clone(),
                    start_time,
                    response,
                ));
            }

            let response = state
                .http_client()
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&chat_req)
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
                    Json(json!({"error": format!("OpenAI Chat API Error: {}", text)})),
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

            state
                .billing_service()
                .record_usage_and_charge(
                    &ctx,
                    UsageBreakdown::from_openai_chat_json(&resp_json),
                    None,
                    Some(start_time.elapsed().as_millis() as i32),
                )
                .await;

            let responses_resp = convert_openai_chat_to_responses(&resp_json);
            return Ok(Json(responses_resp).into_response());
        }
        ResponsesRoute::AnthropicAdapter => {
            let anthropic_request = convert_responses_to_anthropic(&request);
            let mut adapter_request = anthropic_request;
            adapter_request.model = ctx.transport_model_id.clone();

            if request.stream.unwrap_or(false) {
                let stream_result = ctx
                    .adapter
                    .send_message_stream(adapter_request, ctx.provider)
                    .await;

                match stream_result {
                    Ok(stream) => Ok(anthropic_stream_to_responses_sse(
                        state.clone(),
                        ctx.clone(),
                        request.clone(),
                        start_time,
                        stream,
                    )),
                    Err(e) => {
                        tracing::error!(
                            "Service error at {}:{}: {:?}\nBacktrace:\n{}",
                            file!(),
                            line!(),
                            e,
                            Backtrace::capture()
                        );
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

                        let responses_response = convert_anthropic_to_responses(response);
                        Ok(Json(responses_response).into_response())
                    }
                    Err(e) => {
                        tracing::error!(
                            "Service error: {:?}\nStack trace:\n{}",
                            e,
                            Backtrace::capture()
                        );
                        Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({"error": e.to_string()})),
                        )
                            .into_response())
                    }
                }
            }
        }
    }
}
