use crate::types::{ResponseOutputContent, ResponseOutputItem};

pub(crate) fn convert_anthropic_to_responses(
    response: crate::types::AnthropicResponse,
) -> crate::types::ResponsesResponse {
    let mut output: Vec<ResponseOutputItem> = Vec::new();
    let mut message_content: Vec<ResponseOutputContent> = Vec::new();
    let mut reasoning_summaries = Vec::new();

    for block in &response.content {
        match block {
            crate::types::ContentBlock::Text { text, .. } => {
                message_content.push(ResponseOutputContent::OutputText {
                    text: text.clone(),
                    annotations: Some(vec![]),
                });
            }
            crate::types::ContentBlock::Thinking { thinking, .. } => {
                reasoning_summaries.push(crate::types::ResponseReasoningSummary::SummaryText {
                    text: thinking.clone(),
                });
            }
            crate::types::ContentBlock::ToolUse {
                id,
                name,
                input,
                thought_signature,
                ..
            } => {
                output.push(ResponseOutputItem::FunctionCall {
                    id: id.clone(),
                    status: Some("completed".to_string()),
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments: input.to_string(),
                    thought_signature: thought_signature.clone(),
                });
            }
            _ => {}
        }
    }

    if !reasoning_summaries.is_empty() {
        output.insert(
            0,
            ResponseOutputItem::Reasoning {
                id: format!("rs_{}", uuid::Uuid::new_v4()),
                summary: Some(reasoning_summaries),
            },
        );
    }

    if !message_content.is_empty() {
        let insert_at = output
            .iter()
            .position(|item| matches!(item, ResponseOutputItem::FunctionCall { .. }))
            .unwrap_or(output.len());
        output.insert(
            insert_at,
            ResponseOutputItem::Message {
                id: response.id.clone(),
                status: Some("completed".to_string()),
                role: response.role.clone(),
                content: message_content,
            },
        );
    }

    crate::types::ResponsesResponse {
        id: format!("resp_{}", uuid::Uuid::new_v4()),
        object: "response".to_string(),
        model: response.model.clone(),
        created_at: chrono::Utc::now().timestamp(),
        output: Some(output),
        usage: Some(crate::types::ResponseUsage {
            input_tokens: Some(response.usage.input_tokens),
            output_tokens: Some(response.usage.output_tokens),
            total_tokens: Some(response.usage.input_tokens + response.usage.output_tokens),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::convert_anthropic_to_responses;
    use crate::types::{AnthropicResponse, ContentBlock, ResponseOutputItem, Usage};

    #[test]
    fn includes_thought_signature_on_function_call_outputs() {
        let response = AnthropicResponse {
            id: "msg_1".to_string(),
            msg_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "exec_command".to_string(),
                input: serde_json::json!({"cmd":"ls -la"}),
                thought_signature: Some("sig-123".to_string()),
                cache_control: None,
            }],
            model: "gemini-3.1-pro-preview".to_string(),
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: Usage::default(),
        };

        let responses = convert_anthropic_to_responses(response);
        let function_call = responses
            .output
            .unwrap_or_default()
            .into_iter()
            .find_map(|item| match item {
                ResponseOutputItem::FunctionCall {
                    thought_signature, ..
                } => thought_signature,
                _ => None,
            });

        assert_eq!(function_call.as_deref(), Some("sig-123"));
    }
}
