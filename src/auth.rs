use axum::{
    extract::FromRequestParts,
    http::{HeaderMap, StatusCode, request::Parts},
    response::{IntoResponse, Response},
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Authenticated;

#[derive(Clone)]
pub struct AuthClient {
    api_key: String,
}

impl AuthClient {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    pub async fn authenticate(&self, headers: &HeaderMap) -> Result<(), AuthError> {
        if self.api_key.is_empty() {
            return Ok(());
        }

        let token = extract_token(headers).ok_or(AuthError::MissingToken)?;
        if token != self.api_key {
            return Err(AuthError::InvalidToken);
        }

        Ok(())
    }
}

use crate::services::AppState;

impl FromRequestParts<Arc<AppState>> for Authenticated {
    type Rejection = AuthError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        state.auth.authenticate(&parts.headers).await?;
        Ok(Authenticated)
    }
}

fn extract_token(headers: &HeaderMap) -> Option<String> {
    if let Some(header_value) = headers.get("Authorization") {
        header_value
            .to_str()
            .ok()
            .map(|s| s.trim_start_matches("Bearer ").to_string())
    } else if let Some(header_value) = headers.get("x-api-key") {
        header_value.to_str().ok().map(|s| s.to_string())
    } else if let Some(header_value) = headers.get("x-goog-api-key") {
        header_value.to_str().ok().map(|s| s.to_string())
    } else {
        None
    }
}

#[derive(Debug)]
pub enum AuthError {
    MissingToken,
    InvalidToken,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            AuthError::MissingToken => (StatusCode::UNAUTHORIZED, "Missing authentication token"),
            AuthError::InvalidToken => (StatusCode::UNAUTHORIZED, "Invalid authentication token"),
        };
        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::AuthClient;
    use axum::http::{HeaderMap, HeaderValue};

    #[tokio::test]
    async fn accepts_matching_bearer_token() {
        let client = AuthClient::new("secret".to_string());
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", HeaderValue::from_static("Bearer secret"));

        client
            .authenticate(&headers)
            .await
            .expect("matching bearer token should authenticate");
    }

    #[tokio::test]
    async fn accepts_matching_x_api_key() {
        let client = AuthClient::new("secret".to_string());
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("secret"));

        client
            .authenticate(&headers)
            .await
            .expect("matching x-api-key should authenticate");
    }

    #[tokio::test]
    async fn rejects_mismatched_bearer_token() {
        let client = AuthClient::new("secret".to_string());
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", HeaderValue::from_static("Bearer wrong"));

        assert!(client.authenticate(&headers).await.is_err());
    }

    #[tokio::test]
    async fn rejects_missing_token_when_api_key_is_set() {
        let client = AuthClient::new("secret".to_string());
        let headers = HeaderMap::new();

        assert!(client.authenticate(&headers).await.is_err());
    }
}
