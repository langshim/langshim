use crate::handlers::execution::merge_stream_usage;
use crate::services::{AppState, RequestContext, UsageBreakdown, spawn_usage_recording};
use crate::types::{ResponsesRequest, StreamEvent};
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use futures::stream::BoxStream;
use std::backtrace::Backtrace;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

pub(crate) fn anthropic_stream_to_responses_sse(
    state: Arc<AppState>,
    ctx: RequestContext,
    request: ResponsesRequest,
    start_time: Instant,
    stream: BoxStream<'static, Result<StreamEvent, Box<dyn Error + Send + Sync>>>,
) -> Response {
    let sse_stream = async_stream::stream! {
        use futures::StreamExt;
        let mut pinned_stream = stream;
        let mut accumulated_usage = crate::types::Usage::default();
        let mut ttft: Option<i32> = None;
        let response_id = format!("resp_{}", uuid::Uuid::new_v4());
        let created_at = chrono::Utc::now().timestamp();
        let mut sequence_number: u32 = 0;

        let initial_response = serde_json::json!({
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "background": false,
            "completed_at": null,
            "error": null,
            "instructions": request.instructions,
            "model": ctx.transport_model_id,
            "output": [],
            "tools": request.tools,
            "temperature": request.temperature.unwrap_or(1.0),
            "max_output_tokens": request.max_output_tokens,
        });

        yield Ok::<_, axum::Error>(Event::default()
            .event("response.created")
            .json_data(serde_json::json!({
                "type": "response.created",
                "response": initial_response,
                "sequence_number": sequence_number
            }))
            .unwrap());
        sequence_number += 1;

        yield Ok::<_, axum::Error>(Event::default()
            .event("response.in_progress")
            .json_data(serde_json::json!({
                "type": "response.in_progress",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                },
                "sequence_number": sequence_number
            }))
            .unwrap());
        sequence_number += 1;

        let mut output_items: Vec<serde_json::Value> = Vec::new();
        let mut current_item_index: usize = 0;
        let mut message_item_id: Option<String> = None;
        let mut message_output_index: Option<usize> = None;
        let mut reasoning_item_id: Option<String> = None;
        let mut reasoning_output_index: Option<usize> = None;
        let mut message_content_index: usize = 0;
        let mut accumulated_text = String::new();

        enum ActiveBlock {
            Text { item_id: String, output_index: usize, content_index: usize },
            Reasoning { item_id: String, output_index: usize },
            FunctionCall { item_id: String, output_index: usize },
            None,
        }

        let mut active_block = ActiveBlock::None;

        while let Some(result) = pinned_stream.next().await {
            if ttft.is_none() {
                ttft = Some(start_time.elapsed().as_millis() as i32);
            }
            match result {
                Ok(event) => {
                    debug!("Received message: {:?}", event);
                    merge_stream_usage(&mut accumulated_usage, &event, ctx.provider);
                    match &event {
                        StreamEvent::MessageStart { message } => {
                            message_item_id = Some(message.id.clone());
                            accumulated_text.clear();
                            active_block = ActiveBlock::None;
                        }
                        StreamEvent::ContentBlockStart { index: _, content_block } => {
                            accumulated_text.clear();
                            match content_block {
                                crate::types::ContentBlock::Text { .. } => {
                                    let item_id = message_item_id
                                        .clone()
                                        .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4()));
                                    let output_index = if let Some(existing) = message_output_index {
                                        existing
                                    } else {
                                        output_items.push(serde_json::json!({
                                            "type": "message",
                                            "id": item_id,
                                            "status": "in_progress",
                                            "role": "assistant",
                                            "content": [],
                                        }));
                                        let output_index = current_item_index;
                                        yield Ok::<_, axum::Error>(Event::default()
                                            .event("response.output_item.added")
                                            .json_data(serde_json::json!({
                                                "type": "response.output_item.added",
                                                "response_id": response_id,
                                                "output_index": output_index,
                                                "item": output_items[output_index],
                                                "sequence_number": sequence_number
                                            }))
                                            .unwrap());
                                        sequence_number += 1;
                                        current_item_index += 1;
                                        message_output_index = Some(output_index);
                                        output_index
                                    };
                                    if let Some(content) = output_items[output_index]
                                        .get_mut("content")
                                        .and_then(|value| value.as_array_mut())
                                    {
                                        content.push(serde_json::json!({
                                            "type": "output_text",
                                            "text": ""
                                        }));
                                    }
                                    active_block = ActiveBlock::Text {
                                        item_id,
                                        output_index,
                                        content_index: message_content_index,
                                    };
                                    message_content_index += 1;
                                }
                                crate::types::ContentBlock::Thinking { .. } => {
                                    let item_id = reasoning_item_id
                                        .clone()
                                        .unwrap_or_else(|| format!("rs_{}", uuid::Uuid::new_v4()));
                                    let output_index = if let Some(existing) = reasoning_output_index {
                                        existing
                                    } else {
                                        output_items.push(serde_json::json!({
                                            "type": "reasoning",
                                            "id": item_id,
                                            "summary": [],
                                        }));
                                        let output_index = current_item_index;
                                        yield Ok::<_, axum::Error>(Event::default()
                                            .event("response.output_item.added")
                                            .json_data(serde_json::json!({
                                                "type": "response.output_item.added",
                                                "response_id": response_id,
                                                "output_index": output_index,
                                                "item": output_items[output_index],
                                                "sequence_number": sequence_number
                                            }))
                                            .unwrap());
                                        sequence_number += 1;
                                        current_item_index += 1;
                                        reasoning_item_id = Some(item_id.clone());
                                        reasoning_output_index = Some(output_index);
                                        output_index
                                    };
                                    active_block = ActiveBlock::Reasoning { item_id, output_index };
                                }
                                crate::types::ContentBlock::ToolUse {
                                    id,
                                    name,
                                    thought_signature,
                                    ..
                                } => {
                                    let output_index = current_item_index;
                                    output_items.push(serde_json::json!({
                                        "type": "function_call",
                                        "id": id,
                                        "status": "in_progress",
                                        "call_id": id,
                                        "name": name,
                                        "arguments": "",
                                        "thought_signature": thought_signature
                                    }));
                                    debug!(
                                        item = %output_items[output_index],
                                        "Emitting responses function_call output_item.added"
                                    );
                                    yield Ok::<_, axum::Error>(Event::default()
                                        .event("response.output_item.added")
                                        .json_data(serde_json::json!({
                                            "type": "response.output_item.added",
                                            "response_id": response_id,
                                            "output_index": output_index,
                                            "item": output_items[output_index],
                                            "sequence_number": sequence_number
                                        }))
                                        .unwrap());
                                    sequence_number += 1;
                                    current_item_index += 1;
                                    active_block = ActiveBlock::FunctionCall {
                                        item_id: id.clone(),
                                        output_index,
                                    };
                                }
                                _ => {
                                    active_block = ActiveBlock::None;
                                }
                            }
                        }
                        StreamEvent::ContentBlockDelta { index: _, delta } => {
                            match delta {
                                crate::types::ContentBlockDelta::TextDelta { text } => {
                                    if let ActiveBlock::Text { item_id, output_index, content_index } = &active_block {
                                        accumulated_text.push_str(text);
                                        yield Ok::<_, axum::Error>(Event::default()
                                            .event("response.output_text.delta")
                                            .json_data(serde_json::json!({
                                                "type": "response.output_text.delta",
                                                "response_id": response_id,
                                                "content_index": content_index,
                                                "delta": text,
                                                "item_id": item_id,
                                                "logprobs": [],
                                                "output_index": output_index,
                                                "sequence_number": sequence_number
                                            }))
                                            .unwrap());
                                        sequence_number += 1;
                                    }
                                }
                                crate::types::ContentBlockDelta::ThinkingDelta { thinking } => {
                                    if let ActiveBlock::Reasoning { item_id, output_index } = &active_block {
                                        accumulated_text.push_str(thinking);
                                        yield Ok::<_, axum::Error>(Event::default()
                                            .event("response.reasoning.delta")
                                            .json_data(serde_json::json!({
                                                "type": "response.reasoning.delta",
                                                "response_id": response_id,
                                                "delta": thinking,
                                                "item_id": item_id,
                                                "output_index": output_index,
                                                "sequence_number": sequence_number
                                            }))
                                            .unwrap());
                                        sequence_number += 1;
                                    }
                                }
                                crate::types::ContentBlockDelta::InputJsonDelta { partial_json } => {
                                    if let ActiveBlock::FunctionCall { item_id, output_index } = &active_block {
                                        accumulated_text.push_str(partial_json);
                                        yield Ok::<_, axum::Error>(Event::default()
                                            .event("response.function_call_arguments.delta")
                                            .json_data(serde_json::json!({
                                                "type": "response.function_call_arguments.delta",
                                                "response_id": response_id,
                                                "delta": partial_json,
                                                "item_id": item_id,
                                                "output_index": output_index,
                                                "sequence_number": sequence_number
                                            }))
                                            .unwrap());
                                        sequence_number += 1;
                                    }
                                }
                                _ => {}
                            };
                        }
                        StreamEvent::ContentBlockStop { index: _ } => {
                            match &active_block {
                                ActiveBlock::Text { item_id, output_index, content_index } => {
                                    if let Some(content) = output_items[*output_index]
                                        .get_mut("content")
                                        .and_then(|value| value.as_array_mut())
                                    {
                                        if let Some(part) = content.get_mut(*content_index) {
                                            *part = serde_json::json!({
                                                "type": "output_text",
                                                "text": accumulated_text,
                                                "annotations": [],
                                            });
                                        }
                                    }
                                    yield Ok::<_, axum::Error>(Event::default()
                                        .event("response.output_text.done")
                                        .json_data(serde_json::json!({
                                            "type": "response.output_text.done",
                                            "response_id": response_id,
                                            "content_index": content_index,
                                            "item_id": item_id,
                                            "logprobs": [],
                                            "output_index": output_index,
                                            "sequence_number": sequence_number,
                                            "text": accumulated_text
                                        }))
                                        .unwrap());
                                    sequence_number += 1;
                                    yield Ok::<_, axum::Error>(Event::default()
                                        .event("response.content_part.done")
                                        .json_data(serde_json::json!({
                                            "type": "response.content_part.done",
                                            "response_id": response_id,
                                            "content_index": content_index,
                                            "item_id": item_id,
                                            "output_index": output_index,
                                            "part": {
                                                "type": "output_text",
                                                "annotations": [],
                                                "text": accumulated_text
                                            },
                                            "sequence_number": sequence_number
                                        }))
                                        .unwrap());
                                    sequence_number += 1;
                                }
                                ActiveBlock::Reasoning { item_id: _, output_index } => {
                                    if let Some(summary) = output_items[*output_index]
                                        .get_mut("summary")
                                        .and_then(|value| value.as_array_mut())
                                    {
                                        summary.push(serde_json::json!({
                                            "type": "summary_text",
                                            "text": accumulated_text
                                        }));
                                    }
                                }
                                ActiveBlock::FunctionCall { item_id, output_index } => {
                                    output_items[*output_index]["arguments"] = serde_json::json!(accumulated_text);
                                    output_items[*output_index]["status"] = serde_json::json!("completed");
                                    debug!(
                                        item = %output_items[*output_index],
                                        "Emitting responses function_call output_item.done"
                                    );
                                    yield Ok::<_, axum::Error>(Event::default()
                                        .event("response.function_call_arguments.done")
                                        .json_data(serde_json::json!({
                                            "type": "response.function_call_arguments.done",
                                            "response_id": response_id,
                                            "output_index": output_index,
                                            "item": output_items[*output_index],
                                            "sequence_number": sequence_number
                                        }))
                                        .unwrap());
                                    sequence_number += 1;
                                    yield Ok::<_, axum::Error>(Event::default()
                                        .event("response.output_item.done")
                                        .json_data(serde_json::json!({
                                            "type": "response.output_item.done",
                                            "response_id": response_id,
                                            "item": output_items[*output_index],
                                            "output_index": output_index,
                                            "sequence_number": sequence_number
                                        }))
                                        .unwrap());
                                    sequence_number += 1;
                                    let _ = item_id;
                                }
                                ActiveBlock::None => {}
                            }
                            active_block = ActiveBlock::None;
                        }
                        StreamEvent::MessageDelta { .. } => {}
                        StreamEvent::MessageStop => {
                            debug!(
                                output = %serde_json::Value::Array(output_items.clone()),
                                "Emitting responses completed output"
                            );
                            if let Some(output_index) = message_output_index {
                                output_items[output_index]["status"] = serde_json::json!("completed");
                                yield Ok::<_, axum::Error>(Event::default()
                                    .event("response.output_item.done")
                                    .json_data(serde_json::json!({
                                        "type": "response.output_item.done",
                                        "response_id": response_id,
                                        "item": output_items[output_index],
                                        "output_index": output_index,
                                        "sequence_number": sequence_number
                                    }))
                                    .unwrap());
                                sequence_number += 1;
                            }
                            if let Some(output_index) = reasoning_output_index {
                                yield Ok::<_, axum::Error>(Event::default()
                                    .event("response.output_item.done")
                                    .json_data(serde_json::json!({
                                        "type": "response.output_item.done",
                                        "response_id": response_id,
                                        "item": output_items[output_index],
                                        "output_index": output_index,
                                        "sequence_number": sequence_number
                                    }))
                                    .unwrap());
                                sequence_number += 1;
                            }

                            yield Ok::<_, axum::Error>(Event::default()
                                .event("response.completed")
                                .json_data(serde_json::json!({
                                    "type": "response.completed",
                                    "response": {
                                        "id": response_id,
                                        "object": "response",
                                        "created_at": created_at,
                                        "status": "completed",
                                        "background": false,
                                        "completed_at": chrono::Utc::now().timestamp(),
                                        "error": null,
                                        "instructions": request.instructions,
                                        "model": ctx.transport_model_id,
                                        "output": output_items,
                                        "usage": {
                                            "input_tokens": accumulated_usage.input_tokens,
                                            "output_tokens": accumulated_usage.output_tokens,
                                            "total_tokens": accumulated_usage.input_tokens + accumulated_usage.output_tokens,
                                        }
                                    },
                                    "sequence_number": sequence_number
                                }))
                                .unwrap());
                        }
                        StreamEvent::Ping => {
                            yield Ok::<_, axum::Error>(Event::default().event("ping").json_data(serde_json::json!({"type": "ping"})).unwrap());
                        }
                        StreamEvent::Error { error } => {
                            yield Ok::<_, axum::Error>(Event::default()
                                .event("response.error")
                                .json_data(serde_json::json!({
                                    "type": "response.error",
                                    "error": {
                                        "type": error.error_type,
                                        "message": error.message
                                    }
                                }))
                                .unwrap());
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {:?}\nStack trace:\n{}", e, Backtrace::capture());
                    yield Err(axum::Error::new(e));
                }
            }
        }

        spawn_usage_recording(
            state.clone(),
            ctx.clone(),
            UsageBreakdown::from_anthropic_usage(&accumulated_usage),
            ttft,
            Some(start_time.elapsed().as_millis() as i32),
        );

        yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));
    };

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
