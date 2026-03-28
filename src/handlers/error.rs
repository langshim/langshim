use axum::{
    extract::Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::backtrace::Backtrace;

#[derive(Debug)]
pub(crate) struct HandlerError {
    status: StatusCode,
    body: serde_json::Value,
}

impl HandlerError {
    pub(crate) fn new(status: StatusCode, body: serde_json::Value) -> Self {
        Self { status, body }
    }

    pub(crate) fn message(status: StatusCode, message: impl Into<String>) -> Self {
        Self::new(status, json!({ "error": message.into() }))
    }

    pub(crate) fn typed(
        status: StatusCode,
        error_type: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self::new(
            status,
            json!({
                "error": {
                    "type": error_type,
                    "message": message.into(),
                }
            }),
        )
    }
}

impl IntoResponse for HandlerError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

pub(crate) fn internal_error(message: impl Into<String>) -> HandlerError {
    HandlerError::message(StatusCode::INTERNAL_SERVER_ERROR, message)
}

pub(crate) fn log_handler_error(prefix: &str, error: &dyn std::fmt::Debug) {
    tracing::error!(
        "{}: {:?}\nStack trace:\n{}",
        prefix,
        error,
        Backtrace::force_capture()
    );
}

pub(crate) fn log_internal_error_message(prefix: &str, message: impl AsRef<str>) {
    tracing::error!(
        "{}: {}\nStack trace:\n{}",
        prefix,
        message.as_ref(),
        Backtrace::force_capture()
    );
}

pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, HandlerError> {
    serde_json::from_slice(body).map_err(|e| {
        let body_str = String::from_utf8_lossy(body);
        tracing::error!(
            "JSON validation error: {:?}\nPayload: {}\nStack trace:\n{}",
            e,
            body_str,
            Backtrace::force_capture()
        );
        HandlerError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"error": e.to_string()}),
        )
    })
}
