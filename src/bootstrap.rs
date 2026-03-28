use crate::adapter::{
    ModelAdapter, anthropic::AnthropicAdapter, bedrock::BedrockAdapter, gemini::GeminiAdapter,
    openai::OpenAIAdapter,
};
use crate::config::{Config, ConfigOverrides};
use crate::models::{ModelConfigResponse, ModelInfo, TransportProtocol};
use crate::services::{AppState, GatewayClients, ModelRegistry};
use crate::storage::Db;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;

pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_timer(LocalTime)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();
}

#[derive(Clone, Copy, Debug, Default)]
struct LocalTime;

impl FormatTime for LocalTime {
    fn format_time(&self, writer: &mut Writer<'_>) -> fmt::Result {
        writer.write_str(&format_local_timestamp())
    }
}

fn format_local_timestamp() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%.3f%:z")
        .to_string()
}

pub async fn load_config(overrides: ConfigOverrides) -> Config {
    Config::from_sources(overrides)
}

pub async fn build_state(config: Config) -> Arc<AppState> {
    let db = Db::new(&config.data_dir)
        .await
        .expect("Failed to initialize local usage store");

    let openai = OpenAIAdapter::default_converter();
    let http_client = reqwest::Client::new();
    let auth = crate::auth::AuthClient::new(config.api_key.clone());

    let initial_rate = if config.live_exchange {
        let rate = fetch_rate(&http_client).await.unwrap_or_else(|e| {
            tracing::error!("Failed to fetch initial exchange rate: {}", e);
            std::process::exit(1);
        });
        tracing::debug!("Fetched initial USD to CNY rate: {}", rate);
        rate
    } else {
        let default_rate = default_usd_to_cny_rate();
        tracing::info!(
            "Live exchange disabled; using default USD to CNY rate: {}",
            default_rate
        );
        default_rate
    };

    let model_pricing = load_model_config(&config.data_dir);
    let model_currencies = model_pricing
        .iter()
        .map(|(model, info)| (model.clone(), info.currency))
        .collect();
    db.normalize_usage_files(&model_currencies)
        .await
        .expect("Failed to normalize local usage files");
    let adapters = build_adapters(&model_pricing).await;

    Arc::new(AppState {
        db,
        auth,
        clients: GatewayClients {
            openai,
            http: http_client,
        },
        registry: ModelRegistry {
            pricing: model_pricing,
            adapters,
        },
        usd_to_cny_rate: tokio::sync::RwLock::new(initial_rate),
    })
}

pub fn spawn_exchange_rate_updater(state: Arc<AppState>) {
    let state_clone = state.clone();
    let client_clone = state.clients.http.clone();

    tokio::spawn(async move {
        loop {
            use chrono::{Duration, Local};

            let now = Local::now();
            let tomorrow_midnight = (now + Duration::days(1))
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_local_timezone(Local)
                .unwrap();

            let duration_until_midnight = (tomorrow_midnight - now)
                .to_std()
                .unwrap_or(std::time::Duration::from_secs(60));
            tracing::debug!(
                "Scheduling next exchange rate update in {:?}",
                duration_until_midnight
            );
            tokio::time::sleep(duration_until_midnight).await;

            match fetch_rate(&client_clone).await {
                Ok(rate) => {
                    tracing::debug!("Updated USD to CNY rate: {}", rate);
                    *state_clone.usd_to_cny_rate.write().await = rate;
                }
                Err(e) => {
                    tracing::error!("Failed to update exchange rate (keeping old value): {}", e);
                }
            }
        }
    });
}

pub fn default_usd_to_cny_rate() -> Decimal {
    Decimal::new(69, 1)
}

async fn fetch_rate(client: &reqwest::Client) -> Result<Decimal, String> {
    #[derive(Deserialize)]
    struct FrankfurterResponse {
        rates: HashMap<String, Decimal>,
    }

    let response = client
        .get("https://api.frankfurter.app/latest?from=USD&to=CNY")
        .send()
        .await
        .map_err(|err| err.to_string())?;
    let response = response.error_for_status().map_err(|err| err.to_string())?;
    let payload: FrankfurterResponse = response.json().await.map_err(|err| err.to_string())?;

    payload
        .rates
        .get("CNY")
        .copied()
        .ok_or_else(|| "Frankfurter response missing CNY rate".to_string())
}

pub fn load_model_config(data_dir: &std::path::Path) -> HashMap<String, ModelInfo> {
    try_load_model_config(data_dir).unwrap_or_else(|e| {
        tracing::error!("{}", e);
        std::process::exit(1);
    })
}

pub fn try_load_model_config(
    data_dir: &std::path::Path,
) -> Result<HashMap<String, ModelInfo>, String> {
    let model_config_path = data_dir.join("models.json");

    let content = std::fs::read_to_string(&model_config_path)
        .map_err(|e| format!("Failed to read {}: {}", model_config_path.display(), e))?;
    let config_resp = serde_json::from_str::<ModelConfigResponse>(&strip_json_line_comments(&content))
        .map_err(|e| format!("Failed to parse {}: {}", model_config_path.display(), e))?;
    Ok(config_resp.models)
}

fn strip_json_line_comments(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    let mut escaping = false;

    while let Some(ch) = chars.next() {
        if in_string {
            output.push(ch);

            if escaping {
                escaping = false;
                continue;
            }

            match ch {
                '\\' => escaping = true,
                '"' => in_string = false,
                _ => {}
            }

            continue;
        }

        match ch {
            '"' => {
                in_string = true;
                output.push(ch);
            }
            '/' if matches!(chars.peek(), Some('/')) => {
                chars.next();

                while let Some(next) = chars.next() {
                    if next == '\n' {
                        output.push('\n');
                        break;
                    }

                    if next == '\r' {
                        output.push('\r');
                        if matches!(chars.peek(), Some('\n')) {
                            output.push('\n');
                            chars.next();
                        }
                        break;
                    }
                }
            }
            _ => output.push(ch),
        }
    }

    output
}

async fn build_adapters(
    model_pricing: &HashMap<String, ModelInfo>,
) -> HashMap<String, Arc<dyn ModelAdapter + Send + Sync>> {
    let mut adapters: HashMap<String, Arc<dyn ModelAdapter + Send + Sync>> = HashMap::new();

    for (key, model) in model_pricing {
        if let Some(transport) = &model.transport {
            match transport.protocol {
                TransportProtocol::Bedrock => {
                    let adapter = BedrockAdapter::new(
                        transport.base_url.as_deref(),
                        transport.api_key.as_deref(),
                    )
                    .await;
                    adapters.insert(key.clone(), Arc::new(adapter));
                }
                TransportProtocol::Gemini => {
                    if let Some(api_key) = &transport.api_key {
                        let adapter =
                            GeminiAdapter::new(transport.base_url.clone(), api_key.clone());
                        adapters.insert(key.clone(), Arc::new(adapter));
                    }
                }
                TransportProtocol::OpenAI | TransportProtocol::OpenAIResponses => {
                    if let Some(api_key) = &transport.api_key {
                        let adapter =
                            OpenAIAdapter::new(api_key.clone(), transport.base_url.clone());
                        adapters.insert(key.clone(), Arc::new(adapter));
                    }
                }
                TransportProtocol::Anthropic => {
                    if let (Some(url), Some(api_key)) = (&transport.base_url, &transport.api_key) {
                        let adapter = AnthropicAdapter::new(url.clone(), api_key.clone());
                        adapters.insert(key.clone(), Arc::new(adapter));
                    }
                }
            }
        }
    }

    adapters
}

#[cfg(test)]
mod tests {
    use super::{format_local_timestamp, strip_json_line_comments};
    use chrono::Local;

    #[test]
    fn strips_whole_line_comments() {
        let input = "{\n// comment\n\"models\": {}\n}\n";
        let expected = "{\n\n\"models\": {}\n}\n";

        assert_eq!(strip_json_line_comments(input), expected);
    }

    #[test]
    fn strips_inline_comments() {
        let input = "{\n\"models\": {}, // comment\n\"meta\": {}\n}\n";
        let expected = "{\n\"models\": {}, \n\"meta\": {}\n}\n";

        assert_eq!(strip_json_line_comments(input), expected);
    }

    #[test]
    fn keeps_double_slashes_inside_strings() {
        let input = "{\n\"url\": \"https://example.com\", // comment\n\"models\": {}\n}\n";
        let expected = "{\n\"url\": \"https://example.com\", \n\"models\": {}\n}\n";

        assert_eq!(strip_json_line_comments(input), expected);
    }

    #[test]
    fn keeps_escaped_quotes_inside_strings() {
        let input =
            "{\n\"text\": \"\\\\\\\"// not comment\\\\\\\"\", // comment\n\"models\": {}\n}\n";
        let expected = "{\n\"text\": \"\\\\\\\"// not comment\\\\\\\"\", \n\"models\": {}\n}\n";

        assert_eq!(strip_json_line_comments(input), expected);
    }

    #[test]
    fn log_timestamp_uses_system_local_offset() {
        let timestamp = format_local_timestamp();
        let local_offset = Local::now().format("%:z").to_string();

        assert!(timestamp.ends_with(&local_offset));
    }
}
