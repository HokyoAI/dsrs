use super::traits::{Adapter, AdapterConfig};
use super::utils::*;
use crate::primatives::Signature;
use anyhow::{Result, anyhow};
use lazy_static::lazy_static;
use regex::Regex;
use schemars::Schema;
use serde_json::Value as JsonValue;

lazy_static! {
    static ref JSON_PATTERN: Regex = Regex::new(r"\{(?:[^{}]|(?R))*\}").unwrap();
}

pub struct JsonAdapter {
    config: AdapterConfig,
}

impl JsonAdapter {
    pub fn new(config: AdapterConfig) -> Self {
        Self { config }
    }
}

impl<S: Signature> Adapter<S> for JsonAdapter {
    fn config(&self) -> &AdapterConfig {
        &self.config
    }

    fn format_field_description(&self, schema: &Schema) -> String {
        // Similar to ChatAdapter but formatted for JSON mode
        let fields = extract_fields(schema).unwrap_or_default();

        let descriptions: Vec<String> = fields
            .iter()
            .map(|(name, info)| {
                let desc = info.description.as_deref().unwrap_or("No description");
                format!("- {}: {} ({})", name, desc, info.type_name)
            })
            .collect();

        descriptions.join("\n")
    }

    fn format_field_structure(&self, input_schema: &Schema, output_schema: &Schema) -> String {
        let mut parts = Vec::new();
        parts.push("All interactions will be structured in the following way:".to_string());
        parts.push("".to_string());
        parts.push("Input fields:".to_string());
        parts.push(<JsonAdapter as Adapter<S>>::format_field_description(self, input_schema));
        parts.push("".to_string());
        parts.push("Output will be a JSON object with the following fields:".to_string());
        parts.push(<JsonAdapter as Adapter<S>>::format_field_description(self, output_schema));

        parts.join("\n")
    }

    fn format_task_description(&self, instructions: &str) -> String {
        format!("Your task: {}", instructions)
    }

    fn format_user_message_content(&self, inputs: &S::Inputs, schema: &Schema) -> String {
        let fields = extract_fields(schema).unwrap_or_default();
        let json_value = serde_json::to_value(inputs).unwrap_or(JsonValue::Null);

        let mut parts = Vec::new();

        if let JsonValue::Object(map) = json_value {
            for (name, _info) in &fields {
                if let Some(value) = map.get(name) {
                    let formatted = format_value(value);
                    parts.push(format!("{}: {}", name, formatted));
                }
            }
        }

        // Add JSON output requirement
        let output_schema = schemars::schema_for!(S::Outputs);
        let output_fields = extract_fields(&output_schema).unwrap_or_default();
        let field_names: Vec<&str> = output_fields.keys().map(|s| s.as_str()).collect();

        parts.push(format!(
            "\nRespond with a JSON object containing these fields: {}",
            field_names.join(", ")
        ));

        parts.join("\n")
    }

    fn format_assistant_message_content(&self, outputs: &S::Outputs, _schema: &Schema) -> String {
        serde_json::to_string_pretty(outputs).unwrap_or_else(|_| "{}".to_string())
    }

    fn parse(&self, completion: &str, _schema: &Schema) -> Result<S::Outputs> {
        // Extract JSON from completion
        let json_str = if let Some(captures) = JSON_PATTERN.find(completion) {
            captures.as_str()
        } else {
            completion
        };

        serde_json::from_str(json_str).map_err(|e| anyhow!("Failed to parse JSON response: {}", e))
    }
}
