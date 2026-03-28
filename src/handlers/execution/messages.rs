use crate::handlers::error::{HandlerError, internal_error, log_handler_error};
use crate::handlers::execution::merge_stream_usage;
use crate::services::{AppState, RequestContext, UsageBreakdown, spawn_usage_recording};
use crate::types::{AnthropicRequest, StreamEvent};
use axum::{
    extract::Json,
    response::{IntoResponse, Response, Sse, sse::Event},
};
use std::backtrace::Backtrace;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

pub(crate) async fn handle_messages_streaming(
    state: Arc<AppState>,
    ctx: RequestContext,
    start_time: Instant,
    adapter_request: AnthropicRequest,
) -> Result<Response, HandlerError> {
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
                while let Some(result) = pinned_stream.next().await {
                    if ttft.is_none() {
                        ttft = Some(start_time.elapsed().as_millis() as i32);
                    }
                    match result {
                        Ok(event) => {
                            debug!("Received message: {:?}", event);
                            merge_stream_usage(&mut accumulated_usage, &event, ctx.provider);
                            let event_type = match &event {
                                StreamEvent::MessageStart { .. } => "message_start",
                                StreamEvent::ContentBlockStart { .. } => "content_block_start",
                                StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
                                StreamEvent::ContentBlockStop { .. } => "content_block_stop",
                                StreamEvent::MessageDelta { .. } => "message_delta",
                                StreamEvent::MessageStop => "message_stop",
                                StreamEvent::Ping => "ping",
                                StreamEvent::Error { .. } => "error",
                            };
                            yield Ok::<_, axum::Error>(Event::default().event(event_type).json_data(event).unwrap());
                        }
                        Err(e) => {
                            tracing::error!("Stream error: {:?}\nStack trace:\n{}", e, Backtrace::capture());
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
            log_handler_error("Service error", &e);
            Err(internal_error(e.to_string()))
        }
    }
}

pub(crate) async fn handle_messages_non_streaming(
    state: Arc<AppState>,
    ctx: RequestContext,
    start_time: Instant,
    adapter_request: AnthropicRequest,
) -> Result<Response, HandlerError> {
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

            Ok(Json(response).into_response())
        }
        Err(e) => {
            log_handler_error("Service error", &e);
            Err(internal_error(e.to_string()))
        }
    }
}
