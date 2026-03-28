mod chat_completions;
mod count_tokens;
mod error;
mod execution;
mod gemini;
mod messages;
mod responses;
pub(crate) mod translation;

pub use self::chat_completions::handle_chat_completions;
pub use self::count_tokens::handle_count_tokens;
pub(crate) use self::error::HandlerError;
pub use self::gemini::handle_gemini_generate_content;
pub use self::messages::handle_messages;
pub use self::responses::handle_responses;

use axum::response::IntoResponse;

pub async fn health_check() -> impl IntoResponse {
    (axum::http::StatusCode::OK, "OK")
}
