use crate::types::StreamEvent;

pub(crate) fn convert_stream_event_to_openai(
    event: StreamEvent,
    model: String,
) -> Option<serde_json::Value> {
    let id = uuid::Uuid::new_v4().to_string();
    let created = chrono::Utc::now().timestamp();

    match event {
        StreamEvent::MessageStart { message: _ } => Some(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null,
            }]
        })),
        StreamEvent::ContentBlockStart {
            index,
            content_block,
        } => match content_block {
            crate::types::ContentBlock::Text { text, .. } => Some(serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": index,
                    "delta": {"content": text},
                    "finish_reason": null,
                }]
            })),
            crate::types::ContentBlock::Thinking { thinking, .. } => Some(serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": index,
                    "delta": {"content": null, "reasoning_content": thinking},
                    "finish_reason": null,
                }]
            })),
            crate::types::ContentBlock::ToolUse {
                id: tool_id, name, ..
            } => Some(serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": index,
                    "delta": {
                        "tool_calls": [{
                            "index": index,
                            "id": tool_id,
                            "function": {"name": name, "arguments": ""}
                        }]
                    },
                    "finish_reason": null,
                }]
            })),
            _ => None,
        },
        StreamEvent::ContentBlockDelta { index, delta } => match delta {
            crate::types::ContentBlockDelta::TextDelta { text } => Some(serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": index,
                    "delta": {"content": text},
                    "finish_reason": null,
                }]
            })),
            crate::types::ContentBlockDelta::ThinkingDelta { thinking } => {
                Some(serde_json::json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": index,
                        "delta": {"content": null, "reasoning_content": thinking},
                        "finish_reason": null,
                    }]
                }))
            }
            crate::types::ContentBlockDelta::InputJsonDelta { partial_json } => {
                Some(serde_json::json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": index,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "function": {"arguments": partial_json}
                            }]
                        },
                        "finish_reason": null,
                    }]
                }))
            }
            _ => None,
        },
        StreamEvent::ContentBlockStop { index: _ } => None,
        StreamEvent::MessageDelta { delta, usage } => {
            let finish_reason = delta
                .stop_reason
                .as_deref()
                .map(|s| match s {
                    "end_turn" => "stop",
                    "max_tokens" => "length",
                    "tool_use" => "tool_calls",
                    _ => "stop",
                })
                .unwrap_or("stop");

            let total_tokens = usage.input_tokens.unwrap_or(0) + usage.output_tokens;

            Some(serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": usage.input_tokens.unwrap_or(0),
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": total_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": usage.cache_read_input_tokens.unwrap_or(0),
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                    }
                }
            }))
        }
        StreamEvent::MessageStop => None,
        StreamEvent::Ping => None,
        StreamEvent::Error { error } => Some(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "error": {
                "message": error.message,
                "type": error.error_type,
            }
        })),
    }
}
