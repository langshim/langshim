use crate::{
    models::Provider,
    types::{AnthropicRequest, AnthropicResponse, StreamEvent},
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::error::Error;

pub mod anthropic;
pub mod bedrock;
pub mod gemini;
pub mod openai;

#[async_trait]
pub trait ModelAdapter: Send + Sync {
    async fn send_message(
        &self,
        request: AnthropicRequest,
    ) -> Result<AnthropicResponse, Box<dyn Error + Send + Sync>>;

    async fn send_message_stream(
        &self,
        request: AnthropicRequest,
        provider: Provider,
    ) -> Result<
        BoxStream<'static, Result<StreamEvent, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    >;

    async fn count_tokens(
        &self,
        request: AnthropicRequest,
    ) -> Result<u32, Box<dyn Error + Send + Sync>>;
}
