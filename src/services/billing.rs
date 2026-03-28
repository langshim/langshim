use crate::handlers::HandlerError;
use crate::services::{AppState, RequestContext};
use std::sync::Arc;
use tracing::error;

#[derive(Clone, Copy, Debug, Default)]
pub struct UsageBreakdown {
    pub input_tokens: i32,
    pub cache_read_tokens: i32,
    pub cache_creation_tokens: i32,
    pub reasoning_tokens: i32,
    pub output_tokens: i32,
}

impl UsageBreakdown {
    fn has_usage(&self) -> bool {
        self.input_tokens > 0 || self.output_tokens > 0
    }

    fn billable_input_tokens(&self) -> i32 {
        let cached_tokens = self.cache_read_tokens + self.cache_creation_tokens;
        if self.input_tokens >= cached_tokens {
            self.input_tokens - cached_tokens
        } else {
            self.input_tokens
        }
    }

    pub fn from_anthropic_usage(usage: &crate::types::Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens as i32,
            cache_read_tokens: usage.cache_read_input_tokens.unwrap_or(0) as i32,
            cache_creation_tokens: usage.cache_creation_input_tokens.unwrap_or(0) as i32,
            reasoning_tokens: 0,
            output_tokens: usage.output_tokens as i32,
        }
    }

    pub fn from_openai_chat_json(value: &serde_json::Value) -> Self {
        Self {
            input_tokens: value
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            output_tokens: value
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            cache_read_tokens: value
                .get("usage")
                .and_then(|u| u.get("prompt_tokens_details"))
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            cache_creation_tokens: value
                .get("usage")
                .and_then(|u| u.get("prompt_tokens_details"))
                .and_then(|d| d.get("cache_write_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            reasoning_tokens: value
                .get("usage")
                .and_then(|u| u.get("reasoning_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
        }
    }

    pub fn from_responses_json(value: &serde_json::Value) -> Self {
        let usage = value
            .get("response")
            .and_then(|response| response.get("usage"))
            .or_else(|| value.get("usage"));
        Self {
            input_tokens: usage
                .and_then(|u| u.get("input_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            output_tokens: usage
                .and_then(|u| u.get("output_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            cache_read_tokens: usage
                .and_then(|u| u.get("input_tokens_details"))
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            cache_creation_tokens: usage
                .and_then(|u| u.get("input_tokens_details"))
                .and_then(|d| d.get("cache_write_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            reasoning_tokens: usage
                .and_then(|u| u.get("output_tokens_details"))
                .and_then(|d| d.get("reasoning_tokens"))
                .or_else(|| usage.and_then(|u| u.get("reasoning_tokens")))
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
        }
    }
}

pub struct BillingService<'a> {
    state: &'a AppState,
}

impl AppState {
    pub fn billing_service(&self) -> BillingService<'_> {
        BillingService { state: self }
    }
}

impl BillingService<'_> {
    pub async fn ensure_balance(&self) -> Result<(), HandlerError> {
        Ok(())
    }

    pub async fn record_usage_and_charge(
        &self,
        ctx: &RequestContext,
        usage: UsageBreakdown,
        ttft: Option<i32>,
        duration: Option<i32>,
    ) {
        if !usage.has_usage() {
            tracing::warn!(
                "Skip usage logging because parsed usage is empty: {:?}",
                usage
            );
            return;
        }

        let billable_input_tokens = usage.billable_input_tokens();
        let rate = *self.state.usd_to_cny_rate.read().await;
        let (cost, cost_original) = crate::pricing::calculate_cost(
            billable_input_tokens,
            usage.cache_read_tokens,
            usage.cache_creation_tokens,
            usage.reasoning_tokens,
            usage.output_tokens,
            &ctx.model_info,
            rate,
        );

        if let Err(e) = self
            .state
            .db
            .log_usage(
                &ctx.resolved_model,
                ctx.model_info.currency,
                billable_input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_creation_tokens,
                usage.reasoning_tokens,
                cost,
                cost_original,
                ttft,
                duration,
            )
            .await
        {
            error!("Failed to log usage: {:?}", e);
        }
    }
}

pub fn spawn_usage_recording(
    state: Arc<AppState>,
    ctx: RequestContext,
    usage: UsageBreakdown,
    ttft: Option<i32>,
    duration: Option<i32>,
) {
    tokio::spawn(async move {
        state
            .billing_service()
            .record_usage_and_charge(&ctx, usage, ttft, duration)
            .await;
    });
}
