use crate::adapter::ModelAdapter;
use crate::models::Provider;
use crate::types::{
    AnthropicRequest, AnthropicResponse, CacheControl, ErrorDetails, MessageContent, StreamEvent,
};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Client;
use serde_json::Value;
use std::error::Error;
use tracing::debug;

#[derive(Debug, Default)]
struct SseFrameParser {
    buffer: String,
    event: Option<String>,
    data_lines: Vec<String>,
}

#[derive(Debug)]
struct SseFrame {
    event: Option<String>,
    data: String,
}

impl SseFrameParser {
    fn push_chunk(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
        let mut frames = Vec::new();
        self.buffer.push_str(&String::from_utf8_lossy(chunk));

        while let Some(newline_pos) = self.buffer.find('\n') {
            let mut line = self.buffer[..newline_pos].to_string();
            self.buffer.drain(..=newline_pos);

            if line.ends_with('\r') {
                line.pop();
            }

            self.process_line(&line, &mut frames);
        }

        frames
    }

    fn finish(&mut self) -> Vec<SseFrame> {
        let mut frames = Vec::new();

        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            self.process_line(&line, &mut frames);
        }

        if let Some(frame) = self.take_frame() {
            frames.push(frame);
        }

        frames
    }

    fn process_line(&mut self, line: &str, frames: &mut Vec<SseFrame>) {
        if line.is_empty() {
            if let Some(frame) = self.take_frame() {
                frames.push(frame);
            }
            return;
        }

        if line.starts_with(':') {
            return;
        }

        if let Some(rest) = line.strip_prefix("event:") {
            self.event = Some(rest.trim().to_string());
            return;
        }

        if let Some(rest) = line.strip_prefix("data:") {
            self.data_lines.push(rest.trim_start().to_string());
            return;
        }

        if line.starts_with('{') {
            self.data_lines.push(line.to_string());
        }
    }

    fn take_frame(&mut self) -> Option<SseFrame> {
        if self.event.is_none() && self.data_lines.is_empty() {
            return None;
        }

        Some(SseFrame {
            event: self.event.take(),
            data: self.data_lines.drain(..).collect::<Vec<_>>().join("\n"),
        })
    }
}

#[derive(Clone)]
pub struct AnthropicAdapter {
    client: Client,
    base_url: String,
    auth_token: String,
}

impl AnthropicAdapter {
    pub fn new(base_url: String, auth_token: String) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_token,
        }
    }
}

fn log_raw_stream_frame(event_name: Option<&str>, data: &str) {
    match event_name {
        Some(event_name) => debug!(
            event = event_name,
            payload = data,
            "Anthropic raw stream frame"
        ),
        None => debug!(payload = data, "Anthropic raw stream frame"),
    }
}

fn normalize_request_for_anthropic(mut request: AnthropicRequest) -> AnthropicRequest {
    ensure_last_message_cache_control(&mut request);
    request
}

fn ensure_last_message_cache_control(request: &mut AnthropicRequest) {
    let Some(last_message) = request.messages.last_mut() else {
        return;
    };

    match &mut last_message.content {
        MessageContent::String(text) => {
            let text = std::mem::take(text);
            last_message.content = MessageContent::Blocks(vec![crate::types::ContentBlock::Text {
                text,
                cache_control: Some(CacheControl {
                    control_type: "ephemeral".to_string(),
                    ttl: None,
                }),
            }]);
        }
        MessageContent::Blocks(blocks) => {
            let Some(last_block) = blocks.last_mut() else {
                return;
            };

            match last_block {
                crate::types::ContentBlock::Text { cache_control, .. }
                | crate::types::ContentBlock::Image { cache_control, .. }
                | crate::types::ContentBlock::ToolUse { cache_control, .. }
                | crate::types::ContentBlock::ToolResult { cache_control, .. } => {
                    if cache_control.is_none() {
                        *cache_control = Some(CacheControl {
                            control_type: "ephemeral".to_string(),
                            ttl: None,
                        });
                    }
                }
                crate::types::ContentBlock::Thinking { .. } => {}
            }
        }
    }
}

#[async_trait]
impl ModelAdapter for AnthropicAdapter {
    async fn send_message(
        &self,
        request: AnthropicRequest,
    ) -> Result<AnthropicResponse, Box<dyn Error + Send + Sync>> {
        let request = normalize_request_for_anthropic(request);
        let url = format!("{}/v1/messages", self.base_url);

        // print_curl_request(&url, &self.auth_token, &request);

        let response = self
            .client
            .post(&url)
            .header("User-Agent", "claude-cli/2.1.92")
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!(
                "Anthropic API error at {}:{}: {}",
                file!(),
                line!(),
                error_text
            )
            .into());
        }

        let anthropic_response: AnthropicResponse = response.json().await?;
        Ok(anthropic_response)
    }

    async fn send_message_stream(
        &self,
        request: AnthropicRequest,
        _provider: Provider,
    ) -> Result<
        BoxStream<'static, Result<StreamEvent, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    > {
        let request = normalize_request_for_anthropic(request);
        let url = format!("{}/v1/messages", self.base_url);

        // print_curl_request(&url, &self.auth_token, &request);

        let response = self
            .client
            .post(&url)
            .header("User-Agent", "claude-cli/2.1.92")
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!(
                "Anthropic API error at {}:{}: {}",
                file!(),
                line!(),
                error_text
            )
            .into());
        }

        let mut stream = response.bytes_stream();
        let mut parser = SseFrameParser::default();

        let output_stream = async_stream::try_stream! {
            let mut done = false;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        for frame in parser.push_chunk(&bytes) {
                            let event_name = frame.event.as_deref();
                            let data = frame.data.trim();
                            if data.is_empty() {
                                continue;
                            }
                            log_raw_stream_frame(event_name, data);
                            if data == "[DONE]" {
                                done = true;
                                break;
                            }
                            if let Ok(event_json) = serde_json::from_str::<Value>(data) {
                                if let Ok(stream_event) = serde_json::from_value::<StreamEvent>(event_json) {
                                    yield stream_event;
                                }
                            }
                        }

                        if done {
                            break;
                        }
                    }
                    Err(e) => {
                        yield StreamEvent::Error {
                            error: ErrorDetails {
                                error_type: "stream_error".to_string(),
                                message: e.to_string()
                            }
                        };
                    }
                }
            }

            if !done {
                for frame in parser.finish() {
                    let event_name = frame.event.as_deref();
                    let data = frame.data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    log_raw_stream_frame(event_name, data);
                    if let Ok(event_json) = serde_json::from_str::<Value>(data) {
                        if let Ok(stream_event) = serde_json::from_value::<StreamEvent>(event_json) {
                            yield stream_event;
                        }
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
        // Use the actual DeepSeek tokenizer from file
        // Ensure "tokenizer.json" is available in the working directory or provide absolute path
        let tokenizer = tokenizers::Tokenizer::from_file("tokenizer.json")
            .map_err(|e| format!("Failed to load tokenizer.json: {}", e))?;

        // Construct a prompt string that mimics the actual input to the model
        // This is an approximation based on DeepSeek's likely chat template (OpenAI-compatible)
        let mut prompt = String::new();

        // System prompt
        if let Some(system) = &request.system {
            match system {
                crate::types::SystemPrompt::String(s) => {
                    prompt.push_str(s);
                    prompt.push_str("\n\n");
                }
                crate::types::SystemPrompt::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            crate::types::ContentBlock::Text { text, .. } => {
                                prompt.push_str(text);
                                prompt.push_str("\n\n");
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Tools definition (OpenAI format approximation)
        if let Some(tools) = &request.tools {
            // DeepSeek likely sees tools injected into the system prompt or a specific block
            // We'll mimic a standard JSON schema injection
            let tools_json = serde_json::to_string(tools).unwrap_or_default();
            prompt.push_str(&format!("\n# Tools\n\n{}\n\n", tools_json));
        }

        // Messages
        for message in &request.messages {
            // DeepSeek V3 typically uses <|User|> / <|Assistant|> or similar special tokens
            // We'll use a standard chat format that the tokenizer likely handles
            let role_title = match message.role.as_str() {
                "user" => "User",
                "assistant" => "Assistant",
                _ => "User",
            };
            prompt.push_str(&format!("<|{}|>\n", role_title));

            match &message.content {
                crate::types::MessageContent::String(s) => {
                    prompt.push_str(s);
                }
                crate::types::MessageContent::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            crate::types::ContentBlock::Text { text, .. } => {
                                prompt.push_str(text);
                            }
                            crate::types::ContentBlock::Thinking { thinking, .. } => {
                                prompt
                                    .push_str(&format!("<thinking>\n{}\n</thinking>\n", thinking));
                            }
                            crate::types::ContentBlock::ToolUse { name, input, .. } => {
                                // Tool calls are often represented as function call objects in JSON
                                let tool_call_json = serde_json::json!({
                                    "name": name,
                                    "arguments": input
                                });
                                prompt.push_str(&format!(
                                    "<tool_call>\n{}\n</tool_call>\n",
                                    tool_call_json
                                ));
                            }
                            crate::types::ContentBlock::ToolResult { content, .. } => {
                                prompt.push_str("<tool_response>\n");
                                if let Some(content) = content {
                                    match content {
                                        crate::types::ToolResultMessageContent::String(s) => {
                                            prompt.push_str(s);
                                        }
                                        crate::types::ToolResultMessageContent::Blocks(blocks) => {
                                            for b in blocks {
                                                match b {
                                                    crate::types::ToolResultContentBlock::Text { text, .. } => {
                                                        prompt.push_str(text);
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                prompt.push_str("\n</tool_response>\n");
                            }
                            _ => {}
                        }
                    }
                }
            }
            prompt.push_str("\n<|EOT|>\n"); // End of turn marker
        }

        let tokens = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| e.to_string())?;
        Ok(tokens.len() as u32)
    }
}

fn print_curl_request(url: &str, auth_token: &str, request: &AnthropicRequest) {
    if let Ok(body) = serde_json::to_string(request) {
        println!("curl -X POST {} \\", url);
        println!("  -H \"Authorization: Bearer {}\" \\", auth_token);
        println!("  -H \"anthropic-version: 2023-06-01\" \\");
        println!("  -H \"content-type: application/json\" \\");
        println!("  -d '{}'", body);
    }
}

#[cfg(test)]
mod tests {
    use super::{SseFrameParser, ensure_last_message_cache_control};
    use crate::types::{AnthropicRequest, ContentBlock, Message, MessageContent};

    #[test]
    fn parses_sse_frames_split_across_chunks() {
        let mut parser = SseFrameParser::default();

        let frames = parser.push_chunk(b"event: content_block_delta\ndata: {\"type\":\"content_");
        assert!(frames.is_empty());

        let frames = parser.push_chunk(
            b"block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        );
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("content_block_delta"));
        assert_eq!(
            frames[0].data,
            "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}"
        );
    }

    #[test]
    fn flushes_final_frame_without_trailing_blank_line() {
        let mut parser = SseFrameParser::default();

        let frames = parser.push_chunk(b"data: {\"type\":\"message_stop\"}\n");
        assert!(frames.is_empty());

        let frames = parser.finish();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "{\"type\":\"message_stop\"}");
    }

    #[test]
    fn adds_cache_control_to_last_message_when_content_is_string() {
        let mut request = AnthropicRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::String("hello".to_string()),
            }],
            max_tokens: 256,
            metadata: None,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        ensure_last_message_cache_control(&mut request);

        let MessageContent::Blocks(blocks) = &request.messages[0].content else {
            panic!("expected blocks content");
        };
        assert!(matches!(
            blocks.last(),
            Some(ContentBlock::Text { cache_control: Some(crate::types::CacheControl { control_type, .. }), .. })
            if control_type == "ephemeral"
        ));
    }

    #[test]
    fn preserves_existing_cache_control_on_last_message_block() {
        let mut request = AnthropicRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![ContentBlock::Text {
                    text: "hello".to_string(),
                    cache_control: Some(crate::types::CacheControl {
                        control_type: "ephemeral".to_string(),
                        ttl: Some("5m".to_string()),
                    }),
                }]),
            }],
            max_tokens: 256,
            metadata: None,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            cache_control: None,
        };

        ensure_last_message_cache_control(&mut request);

        let MessageContent::Blocks(blocks) = &request.messages[0].content else {
            panic!("expected blocks content");
        };
        assert!(matches!(
            blocks.last(),
            Some(ContentBlock::Text { cache_control: Some(crate::types::CacheControl { ttl: Some(ttl), .. }), .. })
            if ttl == "5m"
        ));
    }
}
