use super::error::{HandlerError, log_handler_error, log_internal_error_message, parse_json};
use super::execution::{forward_sse_with_usage, merge_stream_usage, openai_chat_sse_to_gemini_sse};
use super::translation::{
    convert_anthropic_to_gemini, convert_gemini_to_anthropic,
    convert_gemini_to_openai_chat_request, convert_openai_chat_to_gemini_response,
    convert_stream_event_to_gemini,
};
use crate::auth::Authenticated;
use crate::services::{AppState, GeminiHandlerRoute, UsageBreakdown, spawn_usage_recording};
use crate::types::GeminiGenerateContentRequest;
use axum::{
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response, Sse, sse::Event},
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

pub async fn handle_gemini_generate_content(
    State(state): State<Arc<AppState>>,
    Path(gemini_path): Path<String>,
    _auth: Authenticated,
    body: Bytes,
) -> Result<Response, HandlerError> {
    let (model, is_stream) = parse_gemini_path(&gemini_path)?;

    if is_stream {
        handle_gemini_stream_generate_content_inner(state, model, body).await
    } else {
        handle_gemini_generate_content_inner(state, model, body).await
    }
}

async fn handle_gemini_generate_content_inner(
    state: Arc<AppState>,
    model: String,
    body: Bytes,
) -> Result<Response, HandlerError> {
    let request: GeminiGenerateContentRequest = parse_json(&body)?;
    let start_time = Instant::now();
    let ctx = state.resolve_context(&model)?;

    state.billing_service().ensure_balance().await?;

    match GeminiHandlerRoute::resolve(&ctx) {
        GeminiHandlerRoute::AnthropicAdapter => {
            let mut adapter_request = convert_gemini_to_anthropic(model, false, request);
            adapter_request.model = ctx.transport_model_id.clone();

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

                    Ok(axum::Json(convert_anthropic_to_gemini(response)).into_response())
                }
                Err(error) => {
                    log_handler_error("Gemini service error", &error);
                    Err(HandlerError::message(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        error.to_string(),
                    ))
                }
            }
        }

        GeminiHandlerRoute::GeminiPassthrough => {
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
            let url = format!("{}/v1beta/{}:generateContent", base_url, normalized_model);

            let response = state
                .http_client()
                .post(&url)
                .header("x-goog-api-key", &api_key)
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .map_err(|e| {
                    log_handler_error("Gemini passthrough request error", &e);
                    HandlerError::message(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                log_internal_error_message(
                    "Gemini passthrough upstream error",
                    format!("status={} body={}", status, text),
                );
                return Err(HandlerError::message(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Gemini API Error: {}", text),
                ));
            }

            let resp_json: serde_json::Value = response.json().await.map_err(|e| {
                log_handler_error("Gemini passthrough response decode error", &e);
                HandlerError::message(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?;

            let usage = extract_gemini_usage_from_json(&resp_json);
            state
                .billing_service()
                .record_usage_and_charge(
                    &ctx,
                    usage,
                    None,
                    Some(start_time.elapsed().as_millis() as i32),
                )
                .await;

            Ok(axum::Json(resp_json).into_response())
        }

        GeminiHandlerRoute::OpenAIChatAdapter => {
            let base_url = ctx
                .transport
                .base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1")
                .trim_end_matches('/');
            let api_key = ctx.transport.api_key.as_deref().unwrap_or("").to_string();
            let url = format!("{}/chat/completions", base_url);

            let mut chat_req = convert_gemini_to_openai_chat_request(
                ctx.transport_model_id.clone(),
                false,
                request,
            );
            chat_req.model = ctx.transport_model_id.clone();

            let response = state
                .http_client()
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("content-type", "application/json")
                .json(&chat_req)
                .send()
                .await
                .map_err(|e| {
                    log_handler_error("OpenAI chat request error from Gemini endpoint", &e);
                    HandlerError::message(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                log_internal_error_message(
                    "OpenAI chat upstream error from Gemini endpoint",
                    format!("status={} body={}", status, text),
                );
                return Err(HandlerError::message(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("OpenAI Chat API Error: {}", text),
                ));
            }

            let resp_json: serde_json::Value = response.json().await.map_err(|e| {
                log_handler_error("OpenAI chat response decode error from Gemini endpoint", &e);
                HandlerError::message(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?;

            let usage = UsageBreakdown::from_openai_chat_json(&resp_json);
            state
                .billing_service()
                .record_usage_and_charge(
                    &ctx,
                    usage,
                    None,
                    Some(start_time.elapsed().as_millis() as i32),
                )
                .await;

            let gemini_resp = convert_openai_chat_to_gemini_response(&resp_json, &model);
            Ok(axum::Json(gemini_resp).into_response())
        }
    }
}

async fn handle_gemini_stream_generate_content_inner(
    state: Arc<AppState>,
    model: String,
    body: Bytes,
) -> Result<Response, HandlerError> {
    let request: GeminiGenerateContentRequest = parse_json(&body)?;
    let start_time = Instant::now();
    let ctx = state.resolve_context(&model)?;

    state.billing_service().ensure_balance().await?;

    match GeminiHandlerRoute::resolve(&ctx) {
        GeminiHandlerRoute::AnthropicAdapter => {
            let mut adapter_request = convert_gemini_to_anthropic(model, true, request);
            adapter_request.model = ctx.transport_model_id.clone();
            debug!("{}", json!(adapter_request));

            match ctx
                .adapter
                .send_message_stream(adapter_request, ctx.provider)
                .await
            {
                Ok(stream) => {
                    let sse_stream = async_stream::stream! {
                        use futures::StreamExt;

                        let mut pinned_stream = stream;
                        let mut accumulated_usage = crate::types::Usage::default();
                        let mut ttft: Option<i32> = None;
                        let mut response_id: Option<String> = None;
                        let mut model_version: Option<String> = None;
                        let mut pending_tool_json = String::new();
                        let mut pending_tool_name: Option<String> = None;
                        let mut pending_tool_id: Option<String> = None;

                        while let Some(result) = pinned_stream.next().await {
                            if ttft.is_none() {
                                ttft = Some(start_time.elapsed().as_millis() as i32);
                            }

                            match result {
                                Ok(event) => {
                                    merge_stream_usage(&mut accumulated_usage, &event, ctx.provider);

                                    if let Some(chunk) = convert_stream_event_to_gemini(
                                        event,
                                        &mut response_id,
                                        &mut model_version,
                                        &mut pending_tool_json,
                                        &mut pending_tool_name,
                                        &mut pending_tool_id,
                                    ) {
                                        yield Ok::<_, axum::Error>(
                                            Event::default().json_data(chunk).unwrap(),
                                        );
                                    }
                                }
                                Err(error) => {
                                    tracing::error!("Gemini stream error: {:?}", error);
                                    yield Err(axum::Error::new(error));
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
                    };

                    Ok(Sse::new(sse_stream)
                        .keep_alive(axum::response::sse::KeepAlive::default())
                        .into_response())
                }
                Err(error) => {
                    log_handler_error("Gemini stream service error", &error);
                    Err(HandlerError::message(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        error.to_string(),
                    ))
                }
            }
        }

        GeminiHandlerRoute::GeminiPassthrough => {
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
            let url = format!(
                "{}/v1beta/{}:streamGenerateContent?alt=sse",
                base_url, normalized_model
            );

            let response = state
                .http_client()
                .post(&url)
                .header("x-goog-api-key", &api_key)
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .map_err(|e| {
                    log_handler_error("Gemini stream passthrough request error", &e);
                    HandlerError::message(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                log_internal_error_message(
                    "Gemini stream passthrough upstream error",
                    format!("status={} body={}", status, text),
                );
                return Err(HandlerError::message(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Gemini API Error: {}", text),
                ));
            }

            // Pass the raw Gemini SSE stream through unchanged, while recording usage
            // from the usageMetadata field that Gemini includes in its final chunk.
            Ok(forward_sse_with_usage(
                state.clone(),
                ctx.clone(),
                start_time,
                response.bytes_stream(),
                false,
                |sse: &serde_json::Value| -> Option<UsageBreakdown> {
                    let meta = sse.get("usageMetadata")?;
                    let input = meta
                        .get("promptTokenCount")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0) as i32;
                    let output = meta
                        .get("candidatesTokenCount")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0) as i32;
                    if input == 0 && output == 0 {
                        return None;
                    }
                    Some(UsageBreakdown {
                        input_tokens: input,
                        output_tokens: output,
                        cache_read_tokens: meta
                            .get("cachedContentTokenCount")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0) as i32,
                        cache_creation_tokens: 0,
                        reasoning_tokens: 0,
                    })
                },
            ))
        }

        GeminiHandlerRoute::OpenAIChatAdapter => {
            let base_url = ctx
                .transport
                .base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1")
                .trim_end_matches('/');
            let api_key = ctx.transport.api_key.as_deref().unwrap_or("").to_string();
            let url = format!("{}/chat/completions", base_url);

            let mut chat_req = convert_gemini_to_openai_chat_request(
                ctx.transport_model_id.clone(),
                true,
                request,
            );
            chat_req.model = ctx.transport_model_id.clone();

            let response = state
                .http_client()
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("content-type", "application/json")
                .json(&chat_req)
                .send()
                .await
                .map_err(|e| {
                    log_handler_error("OpenAI chat stream request error from Gemini endpoint", &e);
                    HandlerError::message(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                log_internal_error_message(
                    "OpenAI chat stream upstream error from Gemini endpoint",
                    format!("status={} body={}", status, text),
                );
                return Err(HandlerError::message(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("OpenAI Chat API Error: {}", text),
                ));
            }

            Ok(openai_chat_sse_to_gemini_sse(
                state, ctx, start_time, response,
            ))
        }
    }
}

fn extract_gemini_usage_from_json(json: &serde_json::Value) -> UsageBreakdown {
    let meta = match json.get("usageMetadata") {
        Some(m) => m,
        None => return UsageBreakdown::default(),
    };
    UsageBreakdown {
        input_tokens: meta
            .get("promptTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        output_tokens: meta
            .get("candidatesTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        cache_read_tokens: meta
            .get("cachedContentTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
    }
}

fn parse_gemini_path(gemini_path: &str) -> Result<(String, bool), HandlerError> {
    if let Some(model) = gemini_path.strip_suffix(":streamGenerateContent") {
        return Ok((model.to_string(), true));
    }

    if let Some(model) = gemini_path.strip_suffix(":generateContent") {
        return Ok((model.to_string(), false));
    }

    Err(HandlerError::typed(
        StatusCode::NOT_FOUND,
        "not_found",
        format!("Unsupported Gemini endpoint path: /v1beta/models/{gemini_path}"),
    ))
}
