use crate::adapter::ModelAdapter;
use crate::handlers::translation::{
    GeminiStreamState, convert_anthropic_to_gemini_request, convert_gemini_response_to_anthropic,
    convert_gemini_stream_chunk_to_anthropic,
};
use crate::models::Provider;
use crate::types::{AnthropicRequest, ErrorDetails, GeminiGenerateContentResponse, StreamEvent};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Client;
use serde_json::Value;
use std::error::Error;

#[derive(Debug, Default)]
struct SseFrameParser {
    buffer: String,
    data_lines: Vec<String>,
}

impl SseFrameParser {
    fn push_chunk(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut frames = Vec::new();
        self.buffer.push_str(&String::from_utf8_lossy(chunk));

        while let Some(newline_pos) = self.buffer.find('\n') {
            let mut line = self.buffer[..newline_pos].to_string();
            self.buffer.drain(..=newline_pos);

            if line.ends_with('\r') {
                line.pop();
            }

            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    frames.push(self.data_lines.drain(..).collect::<Vec<_>>().join("\n"));
                }
                continue;
            }

            if let Some(rest) = line.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }

        frames
    }

    fn finish(&mut self) -> Vec<String> {
        let mut frames = Vec::new();

        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            if let Some(rest) = line.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }

        if !self.data_lines.is_empty() {
            frames.push(self.data_lines.drain(..).collect::<Vec<_>>().join("\n"));
        }

        frames
    }
}

#[derive(Clone)]
pub struct GeminiAdapter {
    client: Client,
    base_url: String,
    api_key: String,
}

impl GeminiAdapter {
    pub fn new(base_url: Option<String>, api_key: String) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string())
                .trim_end_matches('/')
                .to_string(),
            api_key,
        }
    }

    fn normalize_model_name(model: &str) -> String {
        if model.starts_with("models/") {
            model.to_string()
        } else {
            format!("models/{model}")
        }
    }

    fn endpoint_url(&self, model: &str, method: &str) -> String {
        format!(
            "{}/v1beta/{}:{}",
            self.base_url,
            Self::normalize_model_name(model),
            method
        )
    }
}

#[async_trait]
impl ModelAdapter for GeminiAdapter {
    async fn send_message(
        &self,
        request: AnthropicRequest,
    ) -> Result<crate::types::AnthropicResponse, Box<dyn Error + Send + Sync>> {
        let model = request.model.clone();
        let gemini_request = convert_anthropic_to_gemini_request(request);
        let response = self
            .client
            .post(self.endpoint_url(&model, "generateContent"))
            .header("x-goog-api-key", &self.api_key)
            .header("content-type", "application/json")
            .json(&gemini_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Google Gemini API error: {}", error_text).into());
        }

        let gemini_response: GeminiGenerateContentResponse = response.json().await?;
        Ok(convert_gemini_response_to_anthropic(
            &model,
            gemini_response,
        ))
    }

    async fn send_message_stream(
        &self,
        request: AnthropicRequest,
        _provider: Provider,
    ) -> Result<
        BoxStream<'static, Result<StreamEvent, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    > {
        let model = request.model.clone();
        let gemini_request = convert_anthropic_to_gemini_request(request);
        let response = self
            .client
            .post(format!(
                "{}?alt=sse",
                self.endpoint_url(&model, "streamGenerateContent")
            ))
            .header("x-goog-api-key", &self.api_key)
            .header("content-type", "application/json")
            .json(&gemini_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Google Gemini API error: {}", error_text).into());
        }

        let mut parser = SseFrameParser::default();
        let mut stream = response.bytes_stream();
        let output_stream = async_stream::try_stream! {
            let mut state = GeminiStreamState::default();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        for frame in parser.push_chunk(&bytes) {
                            let data = frame.trim();
                            if data.is_empty() {
                                continue;
                            }

                            if data == "[DONE]" {
                                break;
                            }

                            match serde_json::from_str::<GeminiGenerateContentResponse>(data) {
                                Ok(chunk) => {
                                    for event in convert_gemini_stream_chunk_to_anthropic(
                                        &model,
                                        chunk,
                                        &mut state,
                                    ) {
                                        yield event;
                                    }
                                }
                                Err(error) => {
                                    yield StreamEvent::Error {
                                        error: ErrorDetails {
                                            error_type: "stream_parse_error".to_string(),
                                            message: error.to_string(),
                                        }
                                    };
                                }
                            }
                        }
                    }
                    Err(error) => {
                        yield StreamEvent::Error {
                            error: ErrorDetails {
                                error_type: "stream_error".to_string(),
                                message: error.to_string(),
                            }
                        };
                    }
                }
            }

            for frame in parser.finish() {
                let data = frame.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<GeminiGenerateContentResponse>(data) {
                    for event in convert_gemini_stream_chunk_to_anthropic(&model, chunk, &mut state)
                    {
                        yield event;
                    }
                }
            }
        };

        Ok(Box::pin(output_stream))
    }

    async fn count_tokens(
        &self,
        request: AnthropicRequest,
    ) -> Result<u32, Box<dyn Error + Send + Sync>> {
        let model = request.model.clone();
        let gemini_request = convert_anthropic_to_gemini_request(request);
        let response = self
            .client
            .post(self.endpoint_url(&model, "countTokens"))
            .header("x-goog-api-key", &self.api_key)
            .header("content-type", "application/json")
            .json(&serde_json::json!({ "generateContentRequest": gemini_request }))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Google Gemini API error: {}", error_text).into());
        }

        let payload: Value = response.json().await?;
        let total_tokens = payload
            .get("totalTokens")
            .and_then(Value::as_u64)
            .ok_or_else(|| "Google Gemini countTokens response missing totalTokens".to_string())?;

        Ok(total_tokens as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::GeminiAdapter;

    #[test]
    fn normalize_model_name_adds_models_prefix() {
        assert_eq!(
            GeminiAdapter::normalize_model_name("gemini-2.5-flash"),
            "models/gemini-2.5-flash"
        );
        assert_eq!(
            GeminiAdapter::normalize_model_name("models/gemini-2.5-flash"),
            "models/gemini-2.5-flash"
        );
    }
}
