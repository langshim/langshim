use axum::body::Body;
use axum::http::{HeaderMap, Request};
use axum::middleware::Next;
use axum::response::Response;
use axum::{
    Router,
    routing::{get, post},
};
use http_body_util::BodyExt;
use std::sync::Arc;
use tower_cookies::CookieManagerLayer;

use crate::handlers::{self};
use crate::services::AppState;

async fn print_request_body_middleware(
    req: Request<Body>,
    next: Next,
) -> Result<Response, axum::http::StatusCode> {
    let (parts, body) = req.into_parts();
    let request_line = format!("{} {}", parts.method, parts.uri);
    let headers = format_headers(&parts.headers);

    tracing::debug!("Incoming Request: {}", request_line);
    tracing::debug!("Request Headers:\n{}", headers);

    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            tracing::error!("Failed to read body in middleware: {}", e);
            return Err(axum::http::StatusCode::BAD_REQUEST);
        }
    };

    if let Ok(body_str) = std::str::from_utf8(&bytes) {
        tracing::debug!("Raw Request Body: {}", body_str);
    } else {
        tracing::debug!("Raw Request Body is not valid UTF-8");
    }

    let req = Request::from_parts(parts, Body::from(bytes));

    Ok(next.run(req).await)
}

fn format_headers(headers: &HeaderMap) -> String {
    headers
        .iter()
        .map(|(name, value)| {
            let rendered = if is_sensitive_header(name.as_str()) {
                "<redacted>".to_string()
            } else {
                value.to_str().unwrap_or("<non-utf8>").to_string()
            };
            format!("{}: {}", name, rendered)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_sensitive_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization"
            | "proxy-authorization"
            | "x-api-key"
            | "x-goog-api-key"
            | "cookie"
            | "set-cookie"
    )
}

pub fn build_app(state: Arc<AppState>) -> Router {
    let debug_request_body = tracing::enabled!(tracing::Level::DEBUG);

    let mut api_routes = Router::new()
        .route("/v1/messages", post(handlers::handle_messages))
        .route(
            "/v1/messages/count_tokens",
            post(handlers::handle_count_tokens),
        )
        .route(
            "/v1/chat/completions",
            post(handlers::handle_chat_completions),
        )
        .route("/v1/responses", post(handlers::handle_responses))
        .route(
            "/v1beta/models/{*gemini_path}",
            post(handlers::handle_gemini_generate_content),
        )
        .layer(CookieManagerLayer::new())
        .layer(
            tower_http::trace::TraceLayer::new_for_http()
                .make_span_with(
                    tower_http::trace::DefaultMakeSpan::new().level(tracing::Level::INFO),
                )
                .on_response(
                    tower_http::trace::DefaultOnResponse::new().level(tracing::Level::INFO),
                ),
        );

    if debug_request_body {
        tracing::info!("Request body debug logging is enabled because DEBUG logging is active");
        api_routes = api_routes.layer(axum::middleware::from_fn(print_request_body_middleware));
    }

    Router::new()
        .merge(api_routes)
        .route("/healthz", get(handlers::health_check))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::format_headers;
    use axum::http::{HeaderMap, HeaderValue};

    #[test]
    fn format_headers_redacts_sensitive_values() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
        headers.insert("x-api-key", HeaderValue::from_static("top-secret"));
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        let rendered = format_headers(&headers);

        assert!(rendered.contains("authorization: <redacted>"));
        assert!(rendered.contains("x-api-key: <redacted>"));
        assert!(rendered.contains("content-type: application/json"));
    }
}
