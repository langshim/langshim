use chrono::{Datelike, Duration, Local, NaiveDate};
use config::{ConfigOverrides, parse_host, parse_port};
use rust_decimal::Decimal;
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::info;

mod adapter;
mod auth;
mod bootstrap;
mod config;
mod handlers;
mod models;
mod pricing;
mod router;
mod services;
mod shutdown;
mod storage;
mod types;

#[tokio::main]
async fn main() {
    bootstrap::init_tracing();
    match parse_command(std::env::args().skip(1).collect()) {
        Command::Serve(config) => run_server(config).await,
        Command::Doctor(config) => run_doctor(config).await,
        Command::Usage { query, config } => run_usage(query, config).await,
    }
}

enum Command {
    Serve(ConfigOverrides),
    Doctor(ConfigOverrides),
    Usage {
        query: storage::UsageQuery,
        config: ConfigOverrides,
    },
}

fn parse_command(args: Vec<String>) -> Command {
    let parsed = parse_args(&args);
    match parsed.command {
        ParsedCommandKind::Serve => Command::Serve(parsed.config),
        ParsedCommandKind::Doctor => Command::Doctor(parsed.config),
        ParsedCommandKind::Usage(query) => Command::Usage {
            query,
            config: parsed.config,
        },
    }
}

struct ParsedArgs {
    command: ParsedCommandKind,
    config: ConfigOverrides,
}

enum ParsedCommandKind {
    Serve,
    Doctor,
    Usage(storage::UsageQuery),
}

fn parse_args(args: &[String]) -> ParsedArgs {
    let today = Local::now().date_naive();
    let mut config = ConfigOverrides::default();
    let mut granularity = storage::UsageGranularity::Day;
    let mut from = today - Duration::days(14);
    let mut to = today;
    let mut command = ParsedCommandKind::Serve;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "serve" => {
                command = ParsedCommandKind::Serve;
                index += 1;
            }
            "doctor" => {
                command = ParsedCommandKind::Doctor;
                index += 1;
            }
            "usage" => {
                command = ParsedCommandKind::Usage(storage::UsageQuery {
                    from,
                    to,
                    granularity,
                });
                index += 1;
            }
            "--home" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--home"));
                config.data_dir = Some(PathBuf::from(value));
                index += 2;
            }
            "--api-key" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--api-key"));
                config.api_key = Some(value.clone());
                index += 2;
            }
            "--port" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--port"));
                config.port = Some(parse_port(value, "--port"));
                index += 2;
            }
            "--live-exchange" => {
                config.live_exchange = Some(true);
                index += 1;
            }
            "--host" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--host"));
                config.host = Some(parse_host(value, "--host"));
                index += 2;
            }
            "daily" => {
                granularity = storage::UsageGranularity::Day;
                index += 1;
            }
            "monthly" => {
                granularity = storage::UsageGranularity::Month;
                index += 1;
            }
            "--days" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--days"));
                let days: i64 = value
                    .parse()
                    .unwrap_or_else(|_| usage_arg_invalid("--days", value));
                if days <= 0 {
                    usage_arg_invalid("--days", value);
                }
                from = today - Duration::days(days - 1);
                to = today;
                index += 2;
            }
            "--from" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--from"));
                from = parse_date(value, "--from");
                index += 2;
            }
            "--to" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--to"));
                to = parse_date(value, "--to");
                index += 2;
            }
            "--month" => {
                let value = args
                    .get(index + 1)
                    .unwrap_or_else(|| usage_arg_missing("--month"));
                let (month_from, month_to) = parse_month(value);
                from = month_from;
                to = month_to;
                granularity = storage::UsageGranularity::Month;
                index += 2;
            }
            "--help" | "-h" => usage_help(),
            other => {
                eprintln!("Unknown argument: {}", other);
                usage_help();
            }
        }

        if matches!(command, ParsedCommandKind::Usage(_)) {
            command = ParsedCommandKind::Usage(storage::UsageQuery {
                from,
                to,
                granularity,
            });
        }
    }

    ParsedArgs { command, config }
}

fn usage_arg_missing(flag: &str) -> ! {
    eprintln!("Missing value for {}", flag);
    usage_help();
}

fn usage_arg_invalid(flag: &str, value: &str) -> ! {
    eprintln!("Invalid value for {}: {}", flag, value);
    usage_help();
}

fn parse_date(value: &str, flag: &str) -> NaiveDate {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").unwrap_or_else(|_| usage_arg_invalid(flag, value))
}

fn parse_month(value: &str) -> (NaiveDate, NaiveDate) {
    let normalized = format!("{}-01", value);
    let first_day = NaiveDate::parse_from_str(&normalized, "%Y-%m-%d")
        .unwrap_or_else(|_| usage_arg_invalid("--month", value));
    let next_month = if first_day.month() == 12 {
        NaiveDate::from_ymd_opt(first_day.year() + 1, 1, 1).unwrap()
    } else {
        NaiveDate::from_ymd_opt(first_day.year(), first_day.month() + 1, 1).unwrap()
    };
    let last_day = next_month.pred_opt().unwrap();
    (first_day, last_day)
}

fn usage_help() -> ! {
    eprintln!("Usage:");
    eprintln!(
        "  langshim [--home DIR] [--api-key TOKEN] [--host IP] [--port PORT] [--live-exchange] serve"
    );
    eprintln!("  langshim [--home DIR] [--live-exchange] doctor");
    eprintln!(
        "  langshim [--home DIR] [--api-key TOKEN] [--host IP] [--port PORT] usage [daily|monthly] [--days N] [--from YYYY-MM-DD --to YYYY-MM-DD]"
    );
    eprintln!("  langshim usage monthly --month YYYY-MM");
    std::process::exit(2);
}

async fn run_server(overrides: ConfigOverrides) {
    let config = bootstrap::load_config(overrides).await;
    let live_exchange = config.live_exchange;
    let host = config.host;
    let port = config.port;
    let state = bootstrap::build_state(config).await;
    if live_exchange {
        bootstrap::spawn_exchange_rate_updater(state.clone());
    }
    let app = router::build_app(state);

    let addr = SocketAddr::from((host, port));
    info!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown::shutdown_signal())
        .await
        .unwrap();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum DoctorLevel {
    Ok,
    Warn,
    Error,
}

impl DoctorLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

struct DoctorFinding {
    level: DoctorLevel,
    scope: String,
    message: String,
}

impl DoctorFinding {
    fn new(level: DoctorLevel, scope: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            level,
            scope: scope.into(),
            message: message.into(),
        }
    }
}

async fn run_doctor(overrides: ConfigOverrides) {
    let config = config::Config::from_sources(overrides);
    let mut findings = Vec::new();

    findings.push(DoctorFinding::new(
        DoctorLevel::Ok,
        "config",
        format!("data dir: {}", config.data_dir.display()),
    ));
    findings.push(DoctorFinding::new(
        DoctorLevel::Ok,
        "config",
        format!("listen address: {}:{}", config.host, config.port),
    ));
    findings.push(DoctorFinding::new(
        if config.api_key == "secret" {
            DoctorLevel::Warn
        } else {
            DoctorLevel::Ok
        },
        "auth",
        if config.api_key == "secret" {
            "local API key is still the default `secret`".to_string()
        } else {
            "local API key is configured".to_string()
        },
    ));
    findings.push(DoctorFinding::new(
        DoctorLevel::Ok,
        "exchange",
        if config.live_exchange {
            "live exchange enabled; startup and daily refresh will use Frankfurter".to_string()
        } else {
            format!(
                "live exchange disabled; using fixed USD/CNY rate {}",
                bootstrap::default_usd_to_cny_rate()
            )
        },
    ));

    if let Err(err) = std::fs::create_dir_all(&config.data_dir) {
        findings.push(DoctorFinding::new(
            DoctorLevel::Error,
            "data_dir",
            format!("failed to create data dir: {}", err),
        ));
    } else {
        findings.push(DoctorFinding::new(
            DoctorLevel::Ok,
            "data_dir",
            "data dir is accessible".to_string(),
        ));
    }

    let usage_dir = config.data_dir.join("usage");
    if let Err(err) = std::fs::create_dir_all(&usage_dir) {
        findings.push(DoctorFinding::new(
            DoctorLevel::Error,
            "usage",
            format!("failed to create usage dir {}: {}", usage_dir.display(), err),
        ));
    } else {
        findings.push(DoctorFinding::new(
            DoctorLevel::Ok,
            "usage",
            format!("usage dir is accessible: {}", usage_dir.display()),
        ));
    }

    let model_config_path = config.data_dir.join("models.json");
    match bootstrap::try_load_model_config(&config.data_dir) {
        Ok(models) => {
            findings.push(DoctorFinding::new(
                DoctorLevel::Ok,
                "models",
                format!(
                    "loaded {} model definition(s) from {}",
                    models.len(),
                    model_config_path.display()
                ),
            ));
            if models.is_empty() {
                findings.push(DoctorFinding::new(
                    DoctorLevel::Warn,
                    "models",
                    "models.json is valid but contains no models".to_string(),
                ));
            }

            for (name, model) in &models {
                let scope = format!("model:{name}");
                let Some(transport) = model.transport.as_ref() else {
                    findings.push(DoctorFinding::new(
                        DoctorLevel::Error,
                        scope,
                        "missing transport section".to_string(),
                    ));
                    continue;
                };

                findings.push(DoctorFinding::new(
                    DoctorLevel::Ok,
                    format!("model:{name}"),
                    format!(
                        "provider={} protocol={} model_id={}",
                        transport.provider.as_str(),
                        transport_protocol_label(transport.protocol),
                        transport.model_id
                    ),
                ));

                if transport.model_id.trim().is_empty() {
                    findings.push(DoctorFinding::new(
                        DoctorLevel::Error,
                        format!("model:{name}"),
                        "transport.model_id is empty".to_string(),
                    ));
                }

                match transport.protocol {
                    crate::models::TransportProtocol::Bedrock => {
                        if transport.base_url.as_deref().unwrap_or("").trim().is_empty() {
                            findings.push(DoctorFinding::new(
                                DoctorLevel::Warn,
                                format!("model:{name}"),
                                "bedrock transport.base_url is empty; region inference may fail".to_string(),
                            ));
                        }
                    }
                    crate::models::TransportProtocol::Gemini
                    | crate::models::TransportProtocol::OpenAI
                    | crate::models::TransportProtocol::OpenAIResponses
                    | crate::models::TransportProtocol::Anthropic => {
                        if transport.base_url.as_deref().unwrap_or("").trim().is_empty() {
                            findings.push(DoctorFinding::new(
                                DoctorLevel::Error,
                                format!("model:{name}"),
                                "transport.base_url is missing".to_string(),
                            ));
                        }
                    }
                }

                if transport.api_key.as_deref().unwrap_or("").trim().is_empty() {
                    findings.push(DoctorFinding::new(
                        DoctorLevel::Warn,
                        format!("model:{name}"),
                        "transport.api_key is empty".to_string(),
                    ));
                }
            }
        }
        Err(err) => findings.push(DoctorFinding::new(DoctorLevel::Error, "models", err)),
    }

    let worst = findings
        .iter()
        .map(|finding| finding.level)
        .max()
        .unwrap_or(DoctorLevel::Ok);

    for finding in &findings {
        println!(
            "[{}] {:<18} {}",
            finding.level.as_str(),
            finding.scope,
            finding.message
        );
    }

    let ok_count = findings
        .iter()
        .filter(|finding| finding.level == DoctorLevel::Ok)
        .count();
    let warn_count = findings
        .iter()
        .filter(|finding| finding.level == DoctorLevel::Warn)
        .count();
    let error_count = findings
        .iter()
        .filter(|finding| finding.level == DoctorLevel::Error)
        .count();

    println!();
    println!(
        "doctor summary: {} ok, {} warn, {} error",
        ok_count, warn_count, error_count
    );

    if worst == DoctorLevel::Error {
        std::process::exit(1);
    }
}

fn transport_protocol_label(protocol: crate::models::TransportProtocol) -> &'static str {
    match protocol {
        crate::models::TransportProtocol::Bedrock => "bedrock",
        crate::models::TransportProtocol::Gemini => "gemini",
        crate::models::TransportProtocol::OpenAI => "openai",
        crate::models::TransportProtocol::OpenAIResponses => "openai-responses",
        crate::models::TransportProtocol::Anthropic => "anthropic",
    }
}

async fn run_usage(query: storage::UsageQuery, overrides: ConfigOverrides) {
    let data_dir = config::Config::from_sources(overrides).data_dir;
    let db = storage::Db::new_at_path(data_dir.join("usage"))
        .await
        .unwrap_or_else(|err| panic!("Failed to initialize usage store: {}", err));
    let model_currencies = bootstrap::load_model_config(&data_dir)
        .into_iter()
        .map(|(model, info)| (model, info.currency))
        .collect();
    db.normalize_usage_files(&model_currencies)
        .await
        .unwrap_or_else(|err| panic!("Failed to normalize usage store: {}", err));
    let rows = db
        .summarize_usage(&query, &model_currencies)
        .await
        .unwrap_or_else(|err| panic!("Failed to query usage: {}", err));

    for line in format_usage_table(&rows) {
        println!("{}", line);
    }
}

fn format_usage_table(rows: &[storage::UsageSummary]) -> Vec<String> {
    let mut lines = vec![format!(
        "{:<10} {:<24} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "period",
        "model",
        "requests",
        "input",
        "output",
        "cache_r",
        "cache_w",
        "total",
        "cost($)",
        "cost_orig"
    )];

    let mut total_requests = 0_u32;
    let mut total_input_tokens = 0_i64;
    let mut total_output_tokens = 0_i64;
    let mut total_cache_read_tokens = 0_i64;
    let mut total_cache_creation_tokens = 0_i64;
    let mut total_tokens = 0_i64;
    let mut total_cost = Decimal::ZERO;
    let mut previous_period: Option<&str> = None;

    for row in rows {
        let display_period = if previous_period == Some(row.period.as_str()) {
            ""
        } else {
            row.period.as_str()
        };
        previous_period = Some(row.period.as_str());

        lines.push(format!(
            "{:<10} {:<24} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            display_period,
            row.model,
            row.requests,
            row.input_tokens,
            row.output_tokens,
            row.cache_read_tokens,
            row.cache_creation_tokens,
            row.total_tokens(),
            row.total_cost.round_dp(4),
            format!(
                "{}{}",
                row.currency.symbol(),
                row.total_cost_original.round_dp(4)
            ),
        ));

        total_requests += row.requests;
        total_input_tokens += row.input_tokens;
        total_output_tokens += row.output_tokens;
        total_cache_read_tokens += row.cache_read_tokens;
        total_cache_creation_tokens += row.cache_creation_tokens;
        total_tokens += row.total_tokens();
        total_cost += row.total_cost;
    }

    lines.push(format!(
        "{:-<10} {:-<24} {:-<8} {:-<12} {:-<12} {:-<12} {:-<12} {:-<12} {:-<12} {:-<12}",
        "", "", "", "", "", "", "", "", "", ""
    ));
    lines.push(format!(
        "{:<10} {:<24} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "total",
        "",
        total_requests,
        total_input_tokens,
        total_output_tokens,
        total_cache_read_tokens,
        total_cache_creation_tokens,
        total_tokens,
        total_cost.round_dp(4),
        "",
    ));

    lines
}

#[cfg(test)]
mod tests {
    use super::{Command, format_usage_table, parse_command, transport_protocol_label};
    use crate::models::TransportProtocol;
    use crate::pricing::Currency;
    use crate::storage::UsageSummary;
    use rust_decimal::Decimal;
    use std::str::FromStr;

    #[test]
    fn usage_table_shows_period_once_per_group_and_appends_total() {
        let rows = vec![
            UsageSummary {
                period: "2026-03-27".to_string(),
                model: "kimi-k2.5".to_string(),
                currency: Currency::CNY,
                requests: 2,
                input_tokens: 100,
                output_tokens: 50,
                cache_read_tokens: 10,
                cache_creation_tokens: 5,
                reasoning_tokens: 3,
                total_cost: Decimal::from_str("1.2345").unwrap(),
                total_cost_original: Decimal::from_str("1.2000").unwrap(),
            },
            UsageSummary {
                period: "2026-03-27".to_string(),
                model: "claude-haiku-4.5".to_string(),
                currency: Currency::USD,
                requests: 1,
                input_tokens: 200,
                output_tokens: 80,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                reasoning_tokens: 1,
                total_cost: Decimal::from_str("2.1000").unwrap(),
                total_cost_original: Decimal::from_str("2.0000").unwrap(),
            },
        ];

        let lines = format_usage_table(&rows);

        assert!(lines[1].starts_with("2026-03-27"));
        assert!(lines[2].starts_with("           "));
        assert!(lines.last().unwrap().starts_with("total"));
        assert!(lines.last().unwrap().contains("       3"));
        assert!(lines.last().unwrap().contains("      300"));
        assert!(lines.last().unwrap().contains("      130"));
        assert!(lines.last().unwrap().contains("      445"));
        assert!(lines.last().unwrap().contains("      3.3345"));
        assert!(lines.last().unwrap().ends_with("            "));
        assert!(lines[1].contains("      ¥1.2000"));
        assert!(lines[2].contains("      $2.0000"));
        assert!(lines[1].contains("      165"));
        assert!(lines[2].contains("      280"));
        assert!(lines[0].contains("total"));
        assert!(!lines[0].contains("reasoning"));
    }

    #[test]
    fn parse_command_prefers_cli_values_for_serve() {
        let command = parse_command(vec![
            "--home".to_string(),
            "/tmp/langshim".to_string(),
            "--api-key".to_string(),
            "cli-secret".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            "4321".to_string(),
            "--live-exchange".to_string(),
            "serve".to_string(),
        ]);

        match command {
            Command::Serve(config) => {
                assert_eq!(
                    config.data_dir.unwrap(),
                    std::path::PathBuf::from("/tmp/langshim")
                );
                assert_eq!(config.api_key.unwrap(), "cli-secret");
                assert_eq!(config.host.unwrap().to_string(), "0.0.0.0");
                assert_eq!(config.port.unwrap(), 4321);
                assert_eq!(config.live_exchange, Some(true));
            }
            Command::Usage { .. } => panic!("expected serve command"),
            Command::Doctor(_) => panic!("expected serve command"),
        }
    }

    #[test]
    fn parse_command_supports_doctor() {
        let command = parse_command(vec![
            "--home".to_string(),
            "/tmp/langshim".to_string(),
            "--live-exchange".to_string(),
            "doctor".to_string(),
        ]);

        match command {
            Command::Doctor(config) => {
                assert_eq!(
                    config.data_dir.unwrap(),
                    std::path::PathBuf::from("/tmp/langshim")
                );
                assert_eq!(config.live_exchange, Some(true));
            }
            _ => panic!("expected doctor command"),
        }
    }

    #[test]
    fn parse_command_accepts_global_options_after_usage() {
        let command = parse_command(vec![
            "usage".to_string(),
            "monthly".to_string(),
            "--month".to_string(),
            "2026-03".to_string(),
            "--home".to_string(),
            "/tmp/langshim".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            "5555".to_string(),
        ]);

        match command {
            Command::Usage { query, config } => {
                assert_eq!(query.granularity, crate::storage::UsageGranularity::Month);
                assert_eq!(query.from.to_string(), "2026-03-01");
                assert_eq!(query.to.to_string(), "2026-03-31");
                assert_eq!(
                    config.data_dir.unwrap(),
                    std::path::PathBuf::from("/tmp/langshim")
                );
                assert_eq!(config.host.unwrap().to_string(), "0.0.0.0");
                assert_eq!(config.port.unwrap(), 5555);
            }
            Command::Serve(_) => panic!("expected usage command"),
            Command::Doctor(_) => panic!("expected usage command"),
        }
    }

    #[test]
    fn transport_protocol_label_matches_cli_values() {
        assert_eq!(transport_protocol_label(TransportProtocol::Bedrock), "bedrock");
        assert_eq!(transport_protocol_label(TransportProtocol::Gemini), "gemini");
        assert_eq!(transport_protocol_label(TransportProtocol::OpenAI), "openai");
        assert_eq!(
            transport_protocol_label(TransportProtocol::OpenAIResponses),
            "openai-responses"
        );
        assert_eq!(
            transport_protocol_label(TransportProtocol::Anthropic),
            "anthropic"
        );
    }
}
