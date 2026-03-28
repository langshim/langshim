use super::error::parse_json;
use crate::auth::Authenticated;
use crate::services::AppState;
use crate::types::AnthropicRequest;
use axum::{
    body::Bytes,
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::backtrace::Backtrace;
use std::sync::Arc;

pub async fn handle_count_tokens(
    State(state): State<Arc<AppState>>,
    _auth: Authenticated,
    body: Bytes,
) -> Result<impl IntoResponse, Response> {
    let request: AnthropicRequest = parse_json(&body).map_err(|e| e.into_response())?;

    let ctx = state
        .resolve_context(&request.model)
        .map_err(|e| e.into_response())?;

    let mut adapter_request = request.clone();
    adapter_request.model = ctx.transport_model_id.clone();

    match ctx.adapter.count_tokens(adapter_request).await {
        Ok(count) => Ok(Json(json!({ "input_tokens": count })).into_response()),
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
