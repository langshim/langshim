use crate::services::{AppState, RequestContext, UsageBreakdown, spawn_usage_recording};
use axum::{
    body::Bytes,
    response::{IntoResponse, Response, Sse, sse::Event},
};
use futures::Stream;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

#[derive(Debug, Default)]
struct SseFrameParser {
    buffer: String,
    event_name: Option<String>,
    data_lines: Vec<String>,
}

impl SseFrameParser {
    fn push_chunk(&mut self, chunk: &str) -> Vec<(Option<String>, String)> {
        let mut frames = Vec::new();
        self.buffer.push_str(chunk);

        while let Some(pos) = self.buffer.find('\n') {
            let mut line = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos);

            if line.ends_with('\r') {
                line.pop();
            }

            if line.is_empty() {
                if let Some(frame) = self.finish_frame() {
                    frames.push(frame);
                }
                continue;
            }

            if let Some(rest) = line.strip_prefix("event:") {
                self.event_name = Some(rest.trim_start().to_string());
                continue;
            }

            if let Some(rest) = line.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }

        frames
    }

    fn finish(&mut self) -> Vec<(Option<String>, String)> {
        let mut frames = Vec::new();

        if !self.buffer.is_empty() {
            let mut line = std::mem::take(&mut self.buffer);
            if line.ends_with('\r') {
                line.pop();
            }
            if let Some(rest) = line.strip_prefix("event:") {
                self.event_name = Some(rest.trim_start().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }

        if let Some(frame) = self.finish_frame() {
            frames.push(frame);
        }

        frames
    }

    fn finish_frame(&mut self) -> Option<(Option<String>, String)> {
        if self.data_lines.is_empty() {
            self.event_name = None;
            return None;
        }

        let data = self.data_lines.drain(..).collect::<Vec<_>>().join("\n");
        let event_name = self.event_name.take();
        Some((event_name, data))
    }
}

fn should_forward_done_frame(data: &str, emit_done_sentinel: bool) -> bool {
    data == "[DONE]" && emit_done_sentinel
}

fn should_append_done_sentinel(done_emitted: bool, emit_done_sentinel: bool) -> bool {
    emit_done_sentinel && !done_emitted
}

pub(crate) fn forward_sse_with_usage<S, F>(
    state: Arc<AppState>,
    ctx: RequestContext,
    start_time: Instant,
    stream: S,
    emit_done_sentinel: bool,
    extract_usage: F,
) -> Response
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin + 'static,
    F: Fn(&serde_json::Value) -> Option<UsageBreakdown> + Send + Sync + 'static,
{
    let sse_stream = async_stream::stream! {
        use futures::StreamExt;
        let mut stream = stream;
        let mut usage = UsageBreakdown::default();
        let mut parser = SseFrameParser::default();
        let mut done_emitted = false;

        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    let data_str = String::from_utf8_lossy(&chunk);
                    debug!("{}", data_str);
                    for (event_name, data) in parser.push_chunk(&data_str) {
                        if data == "[DONE]" {
                            done_emitted = true;
                            if should_forward_done_frame(&data, emit_done_sentinel) {
                                yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));
                            }
                            continue;
                        }
                        if let Ok(sse) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(parsed_usage) = extract_usage(&sse) {
                                usage = parsed_usage;
                            }
                        }

                        let mut event = Event::default();
                        if let Some(name) = event_name {
                            event = event.event(name);
                        }
                        yield Ok::<_, axum::Error>(event.data(data));
                    }
                }
                Err(e) => {
                    yield Err(axum::Error::new(e));
                }
            }
        }

        for (event_name, data) in parser.finish() {
            if data == "[DONE]" {
                done_emitted = true;
                if should_forward_done_frame(&data, emit_done_sentinel) {
                    yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));
                }
                continue;
            }

            if let Ok(sse) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(parsed_usage) = extract_usage(&sse) {
                    usage = parsed_usage;
                }
            }

            let mut event = Event::default();
            if let Some(name) = event_name {
                event = event.event(name);
            }
            yield Ok::<_, axum::Error>(event.data(data));
        }

        spawn_usage_recording(
            state.clone(),
            ctx.clone(),
            usage,
            None,
            Some(start_time.elapsed().as_millis() as i32),
        );

        if should_append_done_sentinel(done_emitted, emit_done_sentinel) {
            yield Ok::<_, axum::Error>(Event::default().data("[DONE]"));
        }
    };

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::{should_append_done_sentinel, should_forward_done_frame};

    #[test]
    fn done_frame_is_not_forwarded_for_gemini_passthrough() {
        assert!(!should_forward_done_frame("[DONE]", false));
    }

    #[test]
    fn done_sentinel_is_appended_for_openai_passthrough_when_missing() {
        assert!(should_append_done_sentinel(false, true));
    }
}
