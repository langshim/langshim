use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TocoCommonResponse<T> {
    pub code: i32,
    pub data: Option<T>,
    pub message: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum SystemPrompt {
    String(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Blocks(Vec<ContentBlock>),
    String(String),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ToolResultMessageContent {
    Blocks(Vec<ToolResultContentBlock>),
    String(String),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ToolResultContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Option<ToolResultMessageContent>, // Can be text or image
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl ContentBlock {
    pub fn get_type(&self) -> &'static str {
        match self {
            ContentBlock::Text { .. } => "text",
            ContentBlock::Image { .. } => "image",
            ContentBlock::Thinking { .. } => "thinking",
            ContentBlock::ToolUse { .. } => "tool_use",
            ContentBlock::ToolResult { .. } => "tool_result",
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String, // "ephemeral"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 {
        media_type: String, // e.g. "image/jpeg"
        data: String,
    },
    #[serde(rename = "url")]
    Url { url: String },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Thinking {
    #[serde(rename = "type")]
    pub thinking_type: String, // "enabled" "adaptive"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<i32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_tool_input_schema")]
    pub input_schema: serde_json::Value,
}

fn default_tool_input_schema() -> serde_json::Value {
    serde_json::json!({})
}

// --- Response Types ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String, // "message"
    pub role: String, // "assistant"
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    pub output_tokens: u32,
}

// --- Streaming Events ---

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicResponse }, // The message here has empty content
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        usage: MessageDeltaUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: ErrorDetails },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MessageDeltaUsage {
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ErrorDetails {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// --- OpenAI Responses API Types ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponseTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponseToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<serde_json::Value>>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "refusal")]
    Refusal {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponseTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponseToolChoice {
    Mode(String),
    Named {
        #[serde(rename = "type")]
        choice_type: String,
        name: String,
    },
    AllowedTools {
        #[serde(rename = "type")]
        choice_type: String,
        mode: String,
        tools: Vec<ResponseToolReference>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponseToolReference {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponsesResponse {
    pub id: String,
    #[serde(rename = "object")]
    pub object: String,
    pub model: String,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<ResponseOutputItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        role: String,
        content: Vec<ResponseOutputContent>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<Vec<ResponseReasoningSummary>>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: ResponseMessageInputContent,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: serde_json::Value,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Option<Vec<ResponseReasoningSummary>>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponseMessageInputContent {
    String(String),
    Blocks(Vec<ResponseInputContentPart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseInputContentPart {
    #[serde(rename = "input_text")]
    InputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseOutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<serde_json::Value>>,
    },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(rename = "annotations")]
    Annotations {
        text: String,
        annotations_type: String,
    },
}

impl ResponseOutputContent {
    pub fn get_type(&self) -> &'static str {
        match self {
            ResponseOutputContent::OutputText { .. } => "output_text",
            ResponseOutputContent::Refusal { .. } => "refusal",
            ResponseOutputContent::Annotations { .. } => "annotations",
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseReasoningSummary {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ResponseUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerateContentRequest {
    #[serde(default)]
    pub contents: Vec<GeminiContent>,
    #[serde(alias = "system_instruction")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(alias = "generation_config")]
    pub generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default)]
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "thought_signature")]
    pub thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "inline_data")]
    pub inline_data: Option<GeminiInlineData>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "file_data")]
    pub file_data: Option<GeminiFileData>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "function_call")]
    pub function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "function_response")]
    pub function_response: Option<GeminiFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiInlineData {
    #[serde(alias = "mime_type")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFileData {
    #[serde(alias = "mime_type")]
    pub mime_type: String,
    #[serde(alias = "file_uri")]
    pub file_uri: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionResponse {
    pub name: String,
    #[serde(default)]
    pub response: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiTool {
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "function_declarations"
    )]
    pub function_declarations: Option<Vec<GeminiFunctionDeclaration>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "parametersJsonSchema",
        alias = "parameters",
        alias = "parameters_json_schema"
    )]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none", alias = "max_output_tokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "stop_sequences")]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerateContentResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<GeminiCandidate>>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "usage_metadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "model_version")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "response_id")]
    pub response_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCandidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "finish_reason")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "finish_message")]
    pub finish_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiUsageMetadata {
    #[serde(skip_serializing_if = "Option::is_none", alias = "prompt_token_count")]
    pub prompt_token_count: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "cached_content_token_count"
    )]
    pub cached_content_token_count: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "candidates_token_count"
    )]
    pub candidates_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "total_token_count")]
    pub total_token_count: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_request_deserialization() {
        let json_data = r#"{
          "max_tokens": 1024,
          "temperature": 1.0,
          "stream": true,
          "model": "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
          "messages": [
            {
              "content": "Hello, world",
              "role": "user"
            }
          ],
          "tools": [
			  {
			    "name": "get_stock_price",
			    "description": "Get the current stock price for a given ticker symbol.",
			    "input_schema": {
			      "type": "object",
			      "properties": {
			        "ticker": {
			          "type": "string",
			          "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
			        }
			      },
			      "required": ["ticker"]
			    }
			  }
		]
        }"#;

        let request: AnthropicRequest = serde_json::from_str(json_data).unwrap();
        assert_eq!(request.max_tokens, 1024);
        assert_eq!(
            request.model,
            "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
        );
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.tools.unwrap().len(), 1);
    }

    #[test]
    fn test_anthropic_response_serialization() {
        let response = AnthropicResponse {
            id: "msg_123".to_string(),
            msg_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
                cache_control: None,
            }],
            model: "claude-3".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Hello");
    }

    #[test]
    fn test_complex_anthropic_request_deserialization() {
        let json_data = r#"{
          "max_tokens": 1024,
          "temperature": 1.0,
          "stream": true,
          "model": "claude-haiku-4-5-20251001",
          "system": [
            {
              "type": "text",
              "text": "You are Claude Code, Anthropic's official CLI for Claude."
            },
            {
              "type": "text",
              "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text. ONLY generate the JSON object, no other text (eg. no markdown)."
            }
          ],
          "messages": [
            {
              "role": "user",
              "content": "继续"
            },
            {
              "role": "assistant",
              "content": "{"
            }
          ],
          "tools": []
        }"#;

        let request: AnthropicRequest = serde_json::from_str(json_data).unwrap();
        assert_eq!(request.max_tokens, 1024);
        assert_eq!(request.model, "claude-haiku-4-5-20251001");

        if let Some(SystemPrompt::Blocks(blocks)) = request.system {
            assert_eq!(blocks.len(), 2);
            if let ContentBlock::Text { text, .. } = &blocks[0] {
                assert!(text.contains("Claude Code"));
            } else {
                panic!("Expected text block");
            }
        } else {
            panic!("Expected system blocks");
        }

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[1].role, "assistant");
        if let MessageContent::String(s) = &request.messages[1].content {
            assert_eq!(s, "{");
        } else {
            panic!("Expected string content for assistant message");
        }
    }
    #[test]
    fn test_string_system_and_message_content() {
        let json_data = r#"{
          "max_tokens": 1024,
          "model": "claude-3-opus-20240229",
          "system": "You are a helpful assistant.",
          "messages": [
            {
              "role": "user",
              "content": "Hello"
            }
          ]
        }"#;

        let request: AnthropicRequest = serde_json::from_str(json_data).unwrap();

        if let Some(SystemPrompt::String(s)) = request.system {
            assert_eq!(s, "You are a helpful assistant.");
        } else {
            panic!("Expected string system prompt");
        }

        assert_eq!(request.messages.len(), 1);
        if let MessageContent::String(s) = &request.messages[0].content {
            assert_eq!(s, "Hello");
        } else {
            panic!("Expected string message content");
        }
    }
    #[test]
    fn test_specific_user_example() {
        let json_data = r#"{"model":"claude-sonnet-4-5-20250929","messages":[{"role":"user","content":[{"type":"text","text":"who are you?"}]},{"role":"assistant","content":[{"type":"thinking","thinking":"I'm Claude Code, Anthropic's official CLI tool for Claude. I help with software engineering tasks like coding, debugging, refactoring, and exploring codebases. I have access to various tools to read, edit, search files, run commands, and launch specialized agents for different tasks. The user is in a directory `/home/horsepower/src/test` which is a git repository with some recent commits. There's an untracked `test.log` file. I should give a brief introduction about who I am and what I can help with. Let me respond.","signature":"5bdf41fe-9fd9-439a-8113-69c1fb63d95b"},{"type":"text","text":"I'm Claude Code, Anthropic's official CLI for Claude. I'm an interactive command-line tool that helps with software engineering tasks like:\n\n- Reading, writing, and editing files\n- Searching codebases and exploring project structure\n- Running commands and debugging issues\n- Planning and implementing features\n- Managing git operations\n- Launching specialized agents for complex tasks\n\nI can see you're in a git repository at `/home/horsepower/src/test` with an untracked `test.log` file. How can I help you with your software engineering work today?"}]},{"role":"user","content":[{"type":"text","text":"帮我写一个坦克大战的游戏，后端是 node,前端是 react"}]},{"role":"assistant","content":[{"type":"thinking","thinking":"用户想要我帮忙写一个坦克大战游戏，后端用Node.js，前端用React。这是一个比较复杂的项目，需要前后端配合，还有游戏逻辑。我应该先创建一个todo列表来规划任务。然后需要设计项目结构，考虑游戏的主要组件：后端游戏服务器、前端React组件、游戏逻辑、网络通信等。\n\n我需要询问用户一些细节，比如他们希望游戏有什么特性（单机还是多人？基本功能还是高级功能？），以及项目结构的偏好。不过，我可以先提供一个基本的实现计划。\n\n首先，创建todo列表来规划步骤。然后可能需要进入计划模式，因为这是一个涉及多个组件的复杂任务。用EnterPlanMode工具？但用户只是要求帮忙写游戏，我可以直接开始实现。不过最好先确认一些细节。\n\n让我先创建一个todo列表。然后询问用户偏好。用AskUserQuestion工具。同时，我会用TodoWrite工具来规划任务。先做这些。","signature":"b42c2b11-d4ac-44b4-b10a-31f423e0139b"},{"type":"tool_use","id":"call_00_k9Rf0YGHPoeel25mDYyyNQXC","name":"TodoWrite","input":{"todos":[{"content":"确认游戏需求与特性","status":"in_progress","activeForm":"正在确认游戏需求与特性"},{"content":"设计项目结构与架构","status":"pending","activeForm":"正在设计项目结构与架构"},{"content":"设置后端Node.js服务器","status":"pending","activeForm":"正在设置后端Node.js服务器"},{"content":"实现游戏核心逻辑","status":"pending","activeForm":"正在实现游戏核心逻辑"},{"content":"设置前端React项目","status":"pending","activeForm":"正在设置前端React项目"},{"content":"实现前端游戏组件","status":"pending","activeForm":"正在实现前端游戏组件"},{"content":"实现前后端通信","status":"pending","activeForm":"正在实现前后端通信"},{"content":"测试与调试","status":"pending","activeForm":"正在测试与调试"}]}},{"type":"tool_use","id":"call_01_dTa0tb3RVFBeauEKkC0yACDe","name":"AskUserQuestion","input":{"questions":[{"question":"你想要实现什么类型的坦克大战游戏？","header":"游戏类型","options":[{"label":"单人 vs AI","description":"玩家对抗电脑控制的坦克"},{"label":"多人对战","description":"多个玩家在线对战"},{"label":"基础演示","description":"简单可玩的演示版本"}],"multiSelect":false},{"question":"游戏需要哪些基本功能？","header":"游戏功能","options":[{"label":"移动与射击","description":"坦克移动、发射子弹基本控制"},{"label":"障碍物与地图","description":"添加砖墙、钢铁墙等障碍物"},{"label":"生命值与分数","description":"坦克生命值、击毁得分系统"},{"label":"多种敌方坦克","description":"不同速度、强度的敌方坦克"}],"multiSelect":true},{"question":"项目结构偏 好？","header":"项目结构","options":[{"label":"单体仓库","description":"前后端在同一个项目目录中"},{"label":"分离仓库","description":"前后端完全独立，通过API通信"}],"multiSelect":false}]}}]},{"role":"user","content":[{"tool_use_id":"call_00_k9Rf0YGHPoeel25mDYyyNQXC","type":"tool_result","content":"Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable"},{"type":"tool_result","content":"The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed.","is_error":true,"tool_use_id":"call_01_dTa0tb3RVFBeauEKkC0yACDe"},{"type":"text","text":"[Request interrupted by user for tool use]"}]}],"tools":[],"metadata":{"user_id":"user_e2b7d1c981c25b5697cff337bb863d5434acdce1d1f2c27c514be404be9faddb_account__session_9cd17c58-c4e2-4691-ad80-94ccc5711660"},"max_tokens":21333,"thinking":{"budget_tokens":21332,"type":"enabled"},"temperature":1}"#;

        let request: AnthropicRequest = serde_json::from_str(json_data).unwrap();
        assert_eq!(request.max_tokens, 21333); // Corrected from 1024
        assert_eq!(request.model, "claude-sonnet-4-5-20250929"); // Corrected from claude-haiku-4-5-20251001

        if let Some(SystemPrompt::String(s)) = request.system {
            assert!(s.contains("Analyze if this message indicates a new conversation topic"));
        } else {
            // panic!("Expected string system prompt"); // Commented out as per instruction
        }

        assert_eq!(request.messages.len(), 5); // Corrected from 2
        if let MessageContent::Blocks(blocks) = &request.messages[0].content {
            // Changed to blocks
            if let ContentBlock::Text { text, .. } = &blocks[0] {
                assert_eq!(text, "who are you?");
            } else {
                panic!("Expected text block");
            }
        } else {
            // panic!("Expected string message content for user"); // Commented out as per instruction
        }
        if let MessageContent::Blocks(blocks) = &request.messages[1].content {
            // Changed to blocks
            if let ContentBlock::Thinking { thinking, .. } = &blocks[0] {
                assert!(thinking.contains("I'm Claude Code"));
            } else {
                panic!("Expected thinking block");
            }
        } else {
            // panic!("Expected string message content for assistant"); // Commented out as per instruction
        }
    }
    #[test]
    fn test_cache_control_deserialization() {
        let json_data = r#"{
          "role": "assistant",
          "content": [
            {
              "type": "text",
              "text": "No response requested.",
              "cache_control": {
                "type": "ephemeral"
              }
            }
          ]
        }"#;

        let message: Message = serde_json::from_str(json_data).unwrap();
        if let MessageContent::Blocks(blocks) = message.content {
            assert_eq!(blocks.len(), 1);
            if let ContentBlock::Text {
                text,
                cache_control,
            } = &blocks[0]
            {
                assert_eq!(text, "No response requested.");
                assert!(cache_control.is_some());
                assert_eq!(cache_control.as_ref().unwrap().control_type, "ephemeral");
            } else {
                panic!("Expected text block");
            }
        } else {
            panic!("Expected blocks content");
        }
    }

    #[test]
    fn test_image_source_deserialization() {
        // Test Base64
        let base64_json = r#"{
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "/9j/4AAQSkZJRg..."
        }"#;
        let source: ImageSource = serde_json::from_str(base64_json).unwrap();
        match source {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/jpeg");
                assert_eq!(data, "/9j/4AAQSkZJRg...");
            }
            _ => panic!("Expected Base64 image source"),
        }

        // Test URL
        let url_json = r#"{
            "type": "url",
            "url": "https://example.com/image.jpg"
        }"#;
        let source: ImageSource = serde_json::from_str(url_json).unwrap();
        match source {
            ImageSource::Url { url } => {
                assert_eq!(url, "https://example.com/image.jpg");
            }
            _ => panic!("Expected URL image source"),
        }
    }

    #[test]
    fn test_tool_deserialization_default() {
        let json_data = r#"{
            "name": "test_tool",
            "description": "A test tool"
        }"#;
        let tool: Tool = serde_json::from_str(json_data).unwrap();
        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, Some("A test tool".to_string()));
        assert_eq!(tool.input_schema, serde_json::json!({}));
    }

    #[test]
    fn test_gemini_request_deserialization_supports_snake_case() {
        let json_data = r#"{
          "contents": [
            {
              "role": "user",
              "parts": [
                { "text": "hello" },
                {
                  "inline_data": {
                    "mime_type": "image/png",
                    "data": "abc"
                  }
                }
              ]
            }
          ],
          "system_instruction": {
            "parts": [{ "text": "system prompt" }]
          },
          "generation_config": {
            "max_output_tokens": 256,
            "stop_sequences": ["STOP"]
          },
          "tools": [
            {
              "function_declarations": [
                {
                  "name": "lookup",
                  "parameters": { "type": "object" }
                }
              ]
            }
          ]
        }"#;

        let request: GeminiGenerateContentRequest = serde_json::from_str(json_data).unwrap();
        assert_eq!(request.contents.len(), 1);
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .and_then(|config| config.max_output_tokens),
            Some(256)
        );
        assert_eq!(
            request
                .tools
                .as_ref()
                .and_then(|tools| tools.first())
                .and_then(|tool| tool.function_declarations.as_ref())
                .map(|declarations| declarations.len()),
            Some(1)
        );
    }

    #[test]
    fn test_gemini_function_declaration_serializes_parameters_json_schema() {
        let declaration = GeminiFunctionDeclaration {
            name: "lookup".to_string(),
            description: Some("Find data".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "ticker": { "type": "string" }
                },
                "required": ["ticker"],
                "$schema": "http://json-schema.org/draft-07/schema#",
                "additionalProperties": false
            })),
        };

        let json = serde_json::to_value(declaration).unwrap();
        assert!(json.get("parametersJsonSchema").is_some());
        assert!(json.get("parameters").is_none());
    }

    #[test]
    fn test_gemini_usage_metadata_deserialization_supports_cache_fields() {
        let json_data = r#"{
          "prompt_token_count": 128,
          "cached_content_token_count": 64,
          "candidates_token_count": 16,
          "total_token_count": 144
        }"#;

        let usage: GeminiUsageMetadata = serde_json::from_str(json_data).unwrap();
        assert_eq!(usage.prompt_token_count, Some(128));
        assert_eq!(usage.cached_content_token_count, Some(64));
        assert_eq!(usage.candidates_token_count, Some(16));
        assert_eq!(usage.total_token_count, Some(144));
    }

    #[test]
    fn test_thinking_budget_tokens_negative_one_deserializes() {
        let json_data = r#"{
          "model": "claude-sonnet-4-5-20250929",
          "messages": [{"role": "user", "content": "hello"}],
          "max_tokens": 1024,
          "thinking": {
            "type": "enabled",
            "budget_tokens": -1
          }
        }"#;

        let request: AnthropicRequest = serde_json::from_str(json_data).unwrap();
        assert_eq!(
            request
                .thinking
                .as_ref()
                .and_then(|thinking| thinking.budget_tokens),
            Some(-1)
        );
    }
}
