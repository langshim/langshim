use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::models::ModelInfo;

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum Currency {
    USD,
    CNY,
}

impl Currency {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::USD => "$",
            Self::CNY => "¥",
        }
    }
}

impl Default for Currency {
    fn default() -> Self {
        Self::USD
    }
}

pub fn calculate_cost(
    input_tokens: i32,
    cache_read: i32,
    cache_creation: i32,
    reasoning: i32,
    output_tokens: i32,
    model: &ModelInfo,
    usd_to_cny_rate: Decimal,
) -> (Decimal, Decimal) {
    let million = Decimal::new(1000000, 0);
    let input_tokens_dec = Decimal::from(input_tokens);

    let cache_read_tokens_dec = Decimal::from(cache_read);
    let cache_creation_tokens_dec = Decimal::from(cache_creation);
    let reasoning_tokens_dec = Decimal::from(reasoning);
    // Python: output_tokens = Decimal(output_tokens - reasoning)
    let output_tokens_dec = Decimal::from(output_tokens - reasoning);

    let total_input = input_tokens_dec + cache_read_tokens_dec + cache_creation_tokens_dec;
    let above_threshold = Decimal::from(model.above);

    let (input_price, cache_read_price, cache_creation_price, reasoning_price, output_price) =
        if model.above > 0 && total_input > above_threshold {
            (
                (input_tokens_dec / million) * model.input_pricing_per_mtoken_above,
                (cache_read_tokens_dec / million) * model.cache_read_pricing_per_mtoken_above,
                (cache_creation_tokens_dec / million) * model.cache_write_pricing_per_mtoken_above,
                (reasoning_tokens_dec / million) * model.reasoning_pricing_per_mtoken_above,
                (output_tokens_dec / million) * model.output_pricing_per_mtoken_above,
            )
        } else {
            (
                (input_tokens_dec / million) * model.input_pricing_per_mtoken,
                (cache_read_tokens_dec / million) * model.cache_read_pricing_per_mtoken,
                (cache_creation_tokens_dec / million) * model.cache_write_pricing_per_mtoken,
                (reasoning_tokens_dec / million) * model.reasoning_pricing_per_mtoken,
                (output_tokens_dec / million) * model.output_pricing_per_mtoken,
            )
        };

    let total =
        input_price + cache_read_price + cache_creation_price + reasoning_price + output_price;

    match model.currency {
        Currency::CNY => {
            let cost = if usd_to_cny_rate.is_zero() {
                total
            } else {
                total / usd_to_cny_rate
            };
            (cost, total)
        }
        Currency::USD => (total, total),
    }
}
