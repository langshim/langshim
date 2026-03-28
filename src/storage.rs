use crate::pricing::Currency;
use chrono::{DateTime, Datelike, Local, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::de::Deserializer;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use tokio::fs::{self, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Debug)]
pub enum DbError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidDate(String),
    InvalidDecimal(String),
}

impl Display for DbError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {}", err),
            Self::Json(err) => write!(f, "json error: {}", err),
            Self::InvalidDate(value) => write!(f, "invalid date: {}", value),
            Self::InvalidDecimal(value) => write!(f, "invalid decimal: {}", value),
        }
    }
}

impl std::error::Error for DbError {}

impl From<std::io::Error> for DbError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for DbError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Clone)]
pub struct Db {
    base_dir: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UsageRecord {
    pub timestamp: DateTime<Utc>,
    pub model: String,
    #[serde(default)]
    pub currency: Currency,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_read: i32,
    pub cache_creation: i32,
    pub reasoning: i32,
    pub cost: String,
    pub cost_original: String,
    pub ttft: Option<i32>,
    pub total_time: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct UsageRecordOnDisk {
    timestamp: DateTime<Utc>,
    model: String,
    #[serde(default)]
    currency: Option<Currency>,
    input_tokens: i32,
    output_tokens: i32,
    cache_read: i32,
    cache_creation: i32,
    reasoning: i32,
    cost: String,
    cost_original: String,
    ttft: Option<i32>,
    total_time: Option<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UsageGranularity {
    Day,
    Month,
}

#[derive(Clone, Debug)]
pub struct UsageQuery {
    pub from: NaiveDate,
    pub to: NaiveDate,
    pub granularity: UsageGranularity,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UsageSummary {
    pub period: String,
    pub model: String,
    pub currency: Currency,
    pub requests: u32,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_creation_tokens: i64,
    pub reasoning_tokens: i64,
    pub total_cost: Decimal,
    pub total_cost_original: Decimal,
}

impl UsageSummary {
    pub fn total_tokens(&self) -> i64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_creation_tokens
    }

    fn append_record(&mut self, record: &UsageRecord) -> Result<(), DbError> {
        self.requests += 1;
        self.input_tokens += i64::from(record.input_tokens);
        self.output_tokens += i64::from(record.output_tokens);
        self.cache_read_tokens += i64::from(record.cache_read);
        self.cache_creation_tokens += i64::from(record.cache_creation);
        self.reasoning_tokens += i64::from(record.reasoning);
        self.total_cost += parse_decimal(&record.cost)?;
        self.total_cost_original += parse_decimal(&record.cost_original)?;
        Ok(())
    }
}

impl Db {
    pub async fn new(data_dir: &Path) -> Result<Self, DbError> {
        Self::new_at_path(data_dir.join("usage")).await
    }

    pub async fn new_at_path(base_dir: PathBuf) -> Result<Self, DbError> {
        fs::create_dir_all(&base_dir).await?;
        Ok(Self { base_dir })
    }

    pub async fn normalize_usage_files(
        &self,
        model_currencies: &std::collections::HashMap<String, Currency>,
    ) -> Result<(), DbError> {
        let mut entries = fs::read_dir(&self.base_dir).await?;
        let mut records_by_path: std::collections::BTreeMap<PathBuf, Vec<UsageRecord>> =
            std::collections::BTreeMap::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }

            let records = self.read_records_from_file(&path, model_currencies).await?;
            for record in records {
                let normalized_path =
                    self.file_path_for_date(local_date_for_timestamp(record.timestamp));
                records_by_path
                    .entry(normalized_path)
                    .or_default()
                    .push(record);
            }
            fs::remove_file(path).await?;
        }

        for (path, records) in records_by_path {
            let mut content = String::new();
            for record in records {
                content.push_str(&serde_json::to_string(&record)?);
                content.push('\n');
            }
            fs::write(path, content).await?;
        }

        Ok(())
    }

    pub async fn log_usage(
        &self,
        model: &str,
        currency: Currency,
        input_tokens: i32,
        output_tokens: i32,
        cache_read: i32,
        cache_creation: i32,
        reasoning: i32,
        cost: Decimal,
        cost_original: Decimal,
        ttft: Option<i32>,
        total_time: Option<i32>,
    ) -> Result<(), DbError> {
        let timestamp = Utc::now();
        let path = self.file_path_for_date(local_date_for_timestamp(timestamp));
        let mut line = serde_json::to_string(&UsageRecord {
            timestamp,
            model: model.to_string(),
            currency,
            input_tokens,
            output_tokens,
            cache_read,
            cache_creation,
            reasoning,
            cost: cost.to_string(),
            cost_original: cost_original.to_string(),
            ttft,
            total_time,
        })?;
        line.push('\n');

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        file.write_all(line.as_bytes()).await?;
        Ok(())
    }

    pub async fn summarize_usage(
        &self,
        query: &UsageQuery,
        model_currencies: &std::collections::HashMap<String, Currency>,
    ) -> Result<Vec<UsageSummary>, DbError> {
        if query.from > query.to {
            return Ok(Vec::new());
        }

        let mut current = query.from;
        let mut summaries = std::collections::BTreeMap::<(String, String), UsageSummary>::new();

        while current <= query.to {
            let path = self.file_path_for_date(current);
            if file_exists(&path).await? {
                let records = self.read_records_from_file(&path, model_currencies).await?;
                for record in records {
                    let record_date = local_date_for_timestamp(record.timestamp);
                    if record_date < query.from || record_date > query.to {
                        continue;
                    }

                    let period = match query.granularity {
                        UsageGranularity::Day => record_date.format("%Y-%m-%d").to_string(),
                        UsageGranularity::Month => {
                            format!("{:04}-{:02}", record_date.year(), record_date.month())
                        }
                    };

                    let key = (period.clone(), record.model.clone());
                    let summary = summaries.entry(key).or_insert_with(|| UsageSummary {
                        period,
                        model: record.model.clone(),
                        currency: record.currency,
                        ..UsageSummary::default()
                    });
                    summary.append_record(&record)?;
                }
            }

            current = current
                .succ_opt()
                .ok_or_else(|| DbError::InvalidDate(query.to.to_string()))?;
        }

        Ok(summaries.into_values().collect())
    }

    fn file_path_for_date(&self, date: NaiveDate) -> PathBuf {
        self.base_dir
            .join(format!("{}.jsonl", date.format("%Y-%m-%d")))
    }

    async fn read_records_from_file(
        &self,
        path: &Path,
        model_currencies: &std::collections::HashMap<String, Currency>,
    ) -> Result<Vec<UsageRecord>, DbError> {
        let file = fs::File::open(path).await?;
        let mut lines = BufReader::new(file).lines();
        let mut records = Vec::new();

        while let Some(line) = lines.next_line().await? {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let stream = Deserializer::from_str(trimmed).into_iter::<UsageRecordOnDisk>();
            for record in stream {
                let record = record?;
                let currency = record.currency.unwrap_or_else(|| {
                    model_currencies
                        .get(&record.model)
                        .copied()
                        .unwrap_or_default()
                });

                records.push(UsageRecord {
                    timestamp: record.timestamp,
                    model: record.model,
                    currency,
                    input_tokens: record.input_tokens,
                    output_tokens: record.output_tokens,
                    cache_read: record.cache_read,
                    cache_creation: record.cache_creation,
                    reasoning: record.reasoning,
                    cost: record.cost,
                    cost_original: record.cost_original,
                    ttft: record.ttft,
                    total_time: record.total_time,
                });
            }
        }

        Ok(records)
    }
}

async fn file_exists(path: &Path) -> Result<bool, DbError> {
    match fs::metadata(path).await {
        Ok(meta) => Ok(meta.is_file()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(DbError::Io(err)),
    }
}

fn local_date_for_timestamp(timestamp: DateTime<Utc>) -> NaiveDate {
    timestamp.with_timezone(&Local).date_naive()
}

fn parse_decimal(value: &str) -> Result<Decimal, DbError> {
    value
        .parse()
        .map_err(|_| DbError::InvalidDecimal(value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{Db, UsageGranularity, UsageQuery};
    use crate::pricing::Currency;
    use chrono::{Duration, Local, LocalResult, NaiveDate, TimeZone, Utc};
    use rust_decimal::Decimal;
    use std::collections::HashMap;
    use tokio::fs;

    fn test_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("langshim-{}-{}", name, uuid::Uuid::new_v4()))
    }

    async fn write_record_for_date(
        db: &Db,
        date: NaiveDate,
        cost: &str,
        currency: Currency,
        model: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(&db.base_dir).await?;
        let record = serde_json::json!({
            "timestamp": format!("{}T01:02:03Z", date.format("%Y-%m-%d")),
            "model": model,
            "currency": currency,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read": 10,
            "cache_creation": 5,
            "reasoning": 3,
            "cost": cost,
            "cost_original": cost,
            "ttft": 12,
            "total_time": 40
        });
        let path = db.file_path_for_date(date);
        fs::write(path, format!("{}\n", serde_json::to_string(&record)?)).await?;
        Ok(())
    }

    #[tokio::test]
    async fn writes_usage_into_daily_jsonl_file() -> Result<(), Box<dyn std::error::Error>> {
        let base_dir = test_dir("write");
        let db = Db::new_at_path(base_dir.clone()).await?;

        db.log_usage(
            "gpt-5",
            Currency::USD,
            100,
            50,
            10,
            5,
            3,
            Decimal::new(123, 2),
            Decimal::new(120, 2),
            Some(10),
            Some(20),
        )
        .await?;

        let today = Local::now().date_naive();
        let path = base_dir.join(format!("{}.jsonl", today.format("%Y-%m-%d")));
        let content = fs::read_to_string(path).await?;

        assert!(content.contains("\"model\":\"gpt-5\""));
        assert!(content.contains("\"currency\":\"USD\""));
        assert!(content.contains("\"cost\":\"1.23\""));
        assert!(content.contains("\"cost_original\":\"1.20\""));
        assert!(!content.contains("\"user_id\""));
        assert!(!content.contains("\"provider\""));
        Ok(())
    }

    #[tokio::test]
    async fn summarizes_daily_range_from_multiple_files() -> Result<(), Box<dyn std::error::Error>>
    {
        let base_dir = test_dir("daily-summary");
        let db = Db::new_at_path(base_dir).await?;
        let d1 = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let d2 = NaiveDate::from_ymd_opt(2026, 3, 11).unwrap();

        write_record_for_date(&db, d1, "1.20", Currency::USD, "gpt-4o").await?;
        write_record_for_date(&db, d2, "2.30", Currency::CNY, "gpt-4o-mini").await?;

        let rows = db
            .summarize_usage(
                &UsageQuery {
                    from: d1,
                    to: d2,
                    granularity: UsageGranularity::Day,
                },
                &HashMap::new(),
            )
            .await?;

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].period, "2026-03-10");
        assert_eq!(rows[0].model, "gpt-4o");
        assert_eq!(rows[0].currency, Currency::USD);
        assert_eq!(rows[0].requests, 1);
        assert_eq!(rows[1].period, "2026-03-11");
        assert_eq!(rows[1].model, "gpt-4o-mini");
        assert_eq!(rows[1].currency, Currency::CNY);
        assert_eq!(rows[1].total_cost.to_string(), "2.30");
        Ok(())
    }

    #[tokio::test]
    async fn summarizes_monthly_range() -> Result<(), Box<dyn std::error::Error>> {
        let base_dir = test_dir("monthly-summary");
        let db = Db::new_at_path(base_dir).await?;
        let d1 = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let d2 = NaiveDate::from_ymd_opt(2026, 3, 18).unwrap();
        let d3 = NaiveDate::from_ymd_opt(2026, 4, 1).unwrap();

        write_record_for_date(&db, d1, "1.20", Currency::USD, "gpt-4o").await?;
        write_record_for_date(&db, d2, "2.30", Currency::USD, "gpt-4o").await?;
        write_record_for_date(&db, d3, "3.40", Currency::CNY, "gpt-4o-mini").await?;

        let rows = db
            .summarize_usage(
                &UsageQuery {
                    from: d1,
                    to: d3,
                    granularity: UsageGranularity::Month,
                },
                &HashMap::new(),
            )
            .await?;

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].period, "2026-03");
        assert_eq!(rows[0].model, "gpt-4o");
        assert_eq!(rows[0].currency, Currency::USD);
        assert_eq!(rows[0].requests, 2);
        assert_eq!(rows[0].total_cost.to_string(), "3.50");
        assert_eq!(rows[1].period, "2026-04");
        assert_eq!(rows[1].model, "gpt-4o-mini");
        assert_eq!(rows[1].currency, Currency::CNY);
        assert_eq!(rows[1].requests, 1);
        Ok(())
    }

    #[tokio::test]
    async fn summarizes_legacy_records_with_currency_from_model_config()
    -> Result<(), Box<dyn std::error::Error>> {
        let base_dir = test_dir("legacy-currency");
        let db = Db::new_at_path(base_dir).await?;
        let d1 = NaiveDate::from_ymd_opt(2026, 3, 10).unwrap();
        let record = serde_json::json!({
            "timestamp": format!("{}T01:02:03Z", d1.format("%Y-%m-%d")),
            "user_id": 1,
            "model": "kimi-k2.5",
            "provider": "openai",
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read": 10,
            "cache_creation": 5,
            "reasoning": 3,
            "cost": "1.20",
            "cost_original": "8.64",
            "ttft": 12,
            "total_time": 40
        });
        let path = db.file_path_for_date(d1);
        fs::write(path, format!("{}\n", serde_json::to_string(&record)?)).await?;

        let mut model_currencies = HashMap::new();
        model_currencies.insert("kimi-k2.5".to_string(), Currency::CNY);

        let rows = db
            .summarize_usage(
                &UsageQuery {
                    from: d1,
                    to: d1,
                    granularity: UsageGranularity::Day,
                },
                &model_currencies,
            )
            .await?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].currency, Currency::CNY);
        Ok(())
    }

    #[tokio::test]
    async fn summarizes_concatenated_records_on_a_single_line()
    -> Result<(), Box<dyn std::error::Error>> {
        let base_dir = test_dir("concatenated-line");
        let db = Db::new_at_path(base_dir).await?;
        let d1 = NaiveDate::from_ymd_opt(2026, 4, 4).unwrap();
        let path = db.file_path_for_date(d1);

        let record_1 = serde_json::json!({
            "timestamp": format!("{}T01:02:03Z", d1.format("%Y-%m-%d")),
            "model": "gpt-5.4-nano",
            "currency": Currency::USD,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read": 10,
            "cache_creation": 5,
            "reasoning": 3,
            "cost": "1.20",
            "cost_original": "1.20",
            "ttft": 12,
            "total_time": 40
        });
        let record_2 = serde_json::json!({
            "timestamp": format!("{}T01:02:04Z", d1.format("%Y-%m-%d")),
            "model": "gpt-5.4-nano",
            "currency": Currency::USD,
            "input_tokens": 200,
            "output_tokens": 30,
            "cache_read": 0,
            "cache_creation": 0,
            "reasoning": 0,
            "cost": "0.80",
            "cost_original": "0.80",
            "ttft": null,
            "total_time": 20
        });

        fs::write(
            path,
            format!(
                "{}{}\n",
                serde_json::to_string(&record_1)?,
                serde_json::to_string(&record_2)?
            ),
        )
        .await?;

        let rows = db
            .summarize_usage(
                &UsageQuery {
                    from: d1,
                    to: d1,
                    granularity: UsageGranularity::Day,
                },
                &HashMap::new(),
            )
            .await?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].requests, 2);
        assert_eq!(rows[0].input_tokens, 300);
        assert_eq!(rows[0].output_tokens, 80);
        assert_eq!(rows[0].total_cost.to_string(), "2.00");
        Ok(())
    }

    #[test]
    fn local_date_for_timestamp_uses_system_timezone() {
        let local_now = Local::now();
        let utc_timestamp = match Local.with_ymd_and_hms(2026, 3, 30, 0, 30, 0) {
            LocalResult::Single(dt) => dt.with_timezone(&Utc),
            LocalResult::Ambiguous(dt, _) => dt.with_timezone(&Utc),
            LocalResult::None => (local_now + Duration::days(1)).with_timezone(&Utc),
        };

        assert_eq!(
            super::local_date_for_timestamp(utc_timestamp),
            utc_timestamp.with_timezone(&Local).date_naive()
        );
    }
}
