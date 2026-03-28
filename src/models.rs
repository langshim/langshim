use crate::pricing::Currency;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn default_currency() -> Currency {
    Currency::USD
}

#[derive(Debug, Copy, Deserialize, Serialize, Clone, PartialEq)]
pub enum Provider {
    #[serde(rename = "amazon")]
    Amazon,
    #[serde(rename = "google")]
    Google,
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "litellm")]
    LiteLLM,
    #[serde(rename = "openrouter")]
    OpenRouter,
    #[serde(rename = "deepseek")]
    DeepSeek,
    #[serde(rename = "aliyun")]
    Aliyun,
    #[serde(rename = "moonshot")]
    Moonshot,
    #[serde(rename = "minimax")]
    MiniMax,
    #[serde(rename = "zai")]
    Zai,
    #[serde(rename = "xiaomi")]
    Xiaomi,
}

#[derive(Debug, Copy, Deserialize, Serialize, Clone, PartialEq)]
pub enum TransportProtocol {
    #[serde(rename = "bedrock")]
    Bedrock,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "openai-responses")]
    OpenAIResponses,
    #[serde(rename = "anthropic")]
    Anthropic,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Amazon => "amazon",
            Provider::Google => "google",
            Provider::OpenAI => "openai",
            Provider::LiteLLM => "litellm",
            Provider::OpenRouter => "openrouter",
            Provider::DeepSeek => "deepseek",
            Provider::Aliyun => "aliyun",
            Provider::Moonshot => "moonshot",
            Provider::MiniMax => "minimax",
            Provider::Zai => "zai",
            Provider::Xiaomi => "xiaomi",
        }
    }
}

impl From<&Provider> for String {
    fn from(provider: &Provider) -> Self {
        provider.as_str().to_string()
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfigResponse {
    pub models: HashMap<String, ModelInfo>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Transport {
    #[serde(rename = "provider")]
    pub provider: Provider,
    pub model_id: String,
    pub protocol: TransportProtocol,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelInfo {
    pub transport: Option<Transport>,
    #[serde(default = "default_currency")]
    pub currency: Currency,
    pub input_pricing_per_mtoken: Decimal,
    pub cache_write_pricing_per_mtoken: Decimal,
    pub cache_read_pricing_per_mtoken: Decimal,
    pub reasoning_pricing_per_mtoken: Decimal,
    pub output_pricing_per_mtoken: Decimal,

    #[serde(default)]
    pub above: u32,
    #[serde(default)]
    pub input_pricing_per_mtoken_above: Decimal,
    #[serde(default)]
    pub cache_write_pricing_per_mtoken_above: Decimal,
    #[serde(default)]
    pub cache_read_pricing_per_mtoken_above: Decimal,
    #[serde(default)]
    pub reasoning_pricing_per_mtoken_above: Decimal,
    #[serde(default)]
    pub output_pricing_per_mtoken_above: Decimal,
}
