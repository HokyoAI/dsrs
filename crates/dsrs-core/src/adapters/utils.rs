use anyhow::{Result, anyhow};
use serde::Serialize;
use serde_json::Value as JsonValue;

// Re-export from schema_parser for backward compatibility
pub use super::schema_parser::{FieldInfo, extract_fields_from_schema as extract_fields};


/// Parse a value according to a schema
pub fn parse_value<T: serde::de::DeserializeOwned>(value_str: &str) -> Result<T> {
    // Try to parse as JSON first
    if let Ok(json_value) = serde_json::from_str::<JsonValue>(value_str) {
        serde_json::from_value(json_value)
            .map_err(|e| anyhow!("Failed to deserialize value: {}", e))
    } else {
        // If not valid JSON, try as a string
        serde_json::from_value(JsonValue::String(value_str.to_string()))
            .map_err(|e| anyhow!("Failed to parse as string: {}", e))
    }
}

/// Format a value for display
pub fn format_value<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(JsonValue::String(s)) => s,
        Ok(json) => serde_json::to_string_pretty(&json).unwrap_or_else(|_| "error".to_string()),
        Err(_) => "error".to_string(),
    }
}
