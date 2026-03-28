use super::error::{HandlerError, parse_json};
use super::execution::{handle_messages_non_streaming, handle_messages_streaming};
use crate::auth::Authenticated;
use crate::services::AppState;
use crate::types::AnthropicRequest;
use axum::{body::Bytes, extract::State, response::Response};
use std::sync::Arc;
use std::time::Instant;

pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    _auth: Authenticated,
    body: Bytes,
) -> Result<Response, HandlerError> {
    let request: AnthropicRequest = parse_json(&body)?;
    let start_time = Instant::now();

    let ctx = state.resolve_context(&request.model)?;
    ctx.ensure_endpoint_supported("/v1/messages")?;
    state.billing_service().ensure_balance().await?;

    let mut adapter_request = request.clone();
    adapter_request.model = ctx.transport_model_id.clone();

    if request.stream.unwrap_or(false) {
        handle_messages_streaming(state, ctx, start_time, adapter_request).await
    } else {
        handle_messages_non_streaming(state, ctx, start_time, adapter_request).await
    }
}
