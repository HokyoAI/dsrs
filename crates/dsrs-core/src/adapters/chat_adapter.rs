use super::traits::{Adapter, AdapterConfig};
use super::utils::*;
use crate::primatives::Signature;
use anyhow::{Result, anyhow};
use lazy_static::lazy_static;
use regex::Regex;
use schemars::Schema;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

lazy_static! {
    static ref FIELD_HEADER_PATTERN: Regex = Regex::new(r"\[\[ ## (\w+) ## \]\]").unwrap();
}

pub struct ChatAdapter {
    config: AdapterConfig,
}

impl ChatAdapter {
    pub fn new(config: AdapterConfig) -> Self {
        Self { config }
    }
}

impl<S: Signature> Adapter<S> for ChatAdapter {
    fn config(&self) -> &AdapterConfig {
        &self.config
    }

    fn format_field_description(&self, schema: &Schema) -> String {
        let fields = extract_fields(schema).unwrap_or_default();

        let descriptions: Vec<String> = fields
            .iter()
            .map(|(name, info)| {
                let desc = info.description.as_deref().unwrap_or("No description");
                format!("- {}: {}", name, desc)
            })
            .collect();

        descriptions.join("\n")
    }

    fn format_field_structure(
        &self,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> String {
        let mut parts = Vec::new();
        parts.push("All interactions will be structured in the following way, with the appropriate values filled in.".to_string());

        // Format input fields
        let input_fields = extract_fields(input_schema).unwrap_or_default();
        for (name, info) in &input_fields {
            parts.push(format!("[[ ## {} ## ]]\n{}", name, info.type_name));
        }

        // Format output fields
        let output_fields = extract_fields(output_schema).unwrap_or_default();
        for (name, info) in &output_fields {
            parts.push(format!("[[ ## {} ## ]]\n{}", name, info.type_name));
        }

        parts.push("[[ ## completed ## ]]".to_string());

        parts.join("\n\n")
    }

    fn format_task_description(&self, instructions: &str) -> String {
        let formatted = instructions
            .lines()
            .map(|line| format!("        {}", line))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "In adhering to this structure, your objective is:\n{}",
            formatted
        )
    }

    fn format_user_message_content(&self, inputs: &S::Inputs, schema: &Schema) -> String {
        let fields = extract_fields(schema).unwrap_or_default();
        let json_value = serde_json::to_value(inputs).unwrap_or(JsonValue::Null);

        let mut parts = Vec::new();

        if let JsonValue::Object(map) = json_value {
            for (name, _info) in &fields {
                if let Some(value) = map.get(name) {
                    let formatted = format_value(value);
                    parts.push(format!("[[ ## {} ## ]]\n{}", name, formatted));
                }
            }
        }

        // Add output requirements
        let output_schema = schemars::schema_for!(S::Outputs);
        let output_fields = extract_fields(&output_schema).unwrap_or_default();
        let mut output_req =
            "Respond with the corresponding output fields, starting with the field ".to_string();

        let field_names: Vec<String> = output_fields
            .keys()
            .map(|name| format!("`[[ ## {} ## ]]`", name))
            .collect();

        output_req.push_str(&field_names.join(", then "));
        output_req.push_str(", and then ending with the marker for `[[ ## completed ## ]]`.");
        parts.push(output_req);

        parts.join("\n\n")
    }

    fn format_assistant_message_content(
        &self,
        outputs: &S::Outputs,
        schema: &Schema,
    ) -> String {
        let fields = extract_fields(schema).unwrap_or_default();
        let json_value = serde_json::to_value(outputs).unwrap_or(JsonValue::Null);

        let mut parts = Vec::new();

        if let JsonValue::Object(map) = json_value {
            for (name, _info) in &fields {
                if let Some(value) = map.get(name) {
                    let formatted = format_value(value);
                    parts.push(format!("[[ ## {} ## ]]\n{}", name, formatted));
                }
            }
        }

        parts.push("[[ ## completed ## ]]".to_string());

        parts.join("\n\n")
    }

    fn parse(&self, completion: &str, _schema: &Schema) -> Result<S::Outputs> {
        let mut sections: Vec<(Option<String>, Vec<String>)> = vec![(None, Vec::new())];

        for line in completion.lines() {
            if let Some(captures) = FIELD_HEADER_PATTERN.captures(line.trim()) {
                let header = captures.get(1).unwrap().as_str().to_string();
                let remaining = line[captures.get(0).unwrap().end()..].trim().to_string();

                sections.push((
                    Some(header),
                    if remaining.is_empty() {
                        Vec::new()
                    } else {
                        vec![remaining]
                    },
                ));
            } else {
                sections.last_mut().unwrap().1.push(line.to_string());
            }
        }

        let sections: HashMap<String, String> = sections
            .into_iter()
            .filter_map(|(k, v)| k.map(|key| (key, v.join("\n").trim().to_string())))
            .collect();

        // Build JSON object from sections
        let mut json_obj = serde_json::Map::new();
        for (key, value) in sections {
            if key != "completed" {
                // Try to parse as JSON, otherwise use as string
                let parsed = serde_json::from_str::<JsonValue>(&value)
                    .unwrap_or_else(|_| JsonValue::String(value.to_string()));
                json_obj.insert(key, parsed);
            }
        }

        serde_json::from_value(JsonValue::Object(json_obj))
            .map_err(|e| anyhow!("Failed to deserialize output: {}", e))
    }
}
