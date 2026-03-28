use crate::adapter::{ModelAdapter, openai::OpenAIAdapter};
use crate::handlers::HandlerError;
use crate::models::ModelInfo;
use crate::services::RequestContext;
use crate::storage::Db;
use std::collections::HashMap;
use std::sync::Arc;

pub struct GatewayClients {
    pub openai: OpenAIAdapter,
    pub http: reqwest::Client,
}

pub struct ModelRegistry {
    pub pricing: HashMap<String, ModelInfo>,
    pub adapters: HashMap<String, Arc<dyn ModelAdapter + Send + Sync>>,
}

pub struct AppState {
    pub db: Db,
    pub auth: crate::auth::AuthClient,
    pub clients: GatewayClients,
    pub registry: ModelRegistry,
    pub usd_to_cny_rate: tokio::sync::RwLock<rust_decimal::Decimal>,
}

impl AppState {
    pub fn resolve_context(&self, model_name: &str) -> Result<RequestContext, HandlerError> {
        self.registry.resolve_context(model_name)
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.clients.http
    }

    pub fn openai_adapter(&self) -> &OpenAIAdapter {
        &self.clients.openai
    }
}
