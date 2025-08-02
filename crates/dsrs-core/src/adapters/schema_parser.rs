use anyhow::{Result, anyhow};
use schemars::Schema;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use crate::primatives::Signature;

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub type_name: String,
    pub description: Option<String>,
    pub required: bool,
}

/// Convert a Schema to JSON and extract field information
pub fn extract_fields_from_schema(schema: &Schema) -> Result<HashMap<String, FieldInfo>> {
    // Convert schema to JSON for runtime navigation
    let schema_json = serde_json::to_value(schema)
        .map_err(|e| anyhow!("Failed to serialize schema to JSON: {}", e))?;
    
    extract_fields_from_json(&schema_json)
}

/// Extract field information from a JSON schema representation
pub fn extract_fields_from_json(schema_json: &JsonValue) -> Result<HashMap<String, FieldInfo>> {
    let mut fields = HashMap::new();
    
    // Navigate the JSON schema structure
    if let Some(object_def) = schema_json.get("object") {
        if let Some(properties) = object_def.get("properties").and_then(|p| p.as_object()) {
            // Get required fields
            let required_fields: Vec<String> = object_def
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default();
            
            for (field_name, field_schema) in properties {
                let field_info = extract_field_info_from_json(
                    field_name, 
                    field_schema, 
                    required_fields.contains(field_name)
                )?;
                fields.insert(field_name.clone(), field_info);
            }
        }
    }
    
    Ok(fields)
}

/// Extract information for a single field from JSON schema
fn extract_field_info_from_json(name: &str, field_json: &JsonValue, required: bool) -> Result<FieldInfo> {
    let type_name = extract_type_name_from_json(field_json);
    let description = field_json
        .get("description")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string());
    
    Ok(FieldInfo {
        name: name.to_string(),
        type_name,
        description,
        required,
    })
}

/// Extract type name from JSON schema field
fn extract_type_name_from_json(field_json: &JsonValue) -> String {
    // Check for direct type field
    if let Some(type_value) = field_json.get("type") {
        if let Some(type_str) = type_value.as_str() {
            return match type_str {
                "string" => "String".to_string(),
                "number" => "Number".to_string(),
                "integer" => "Integer".to_string(),
                "boolean" => "Boolean".to_string(),
                "array" => "Array".to_string(),
                "object" => "Object".to_string(),
                "null" => "Null".to_string(),
                _ => type_str.to_string(),
            };
        }
        // Handle array of types
        if let Some(type_array) = type_value.as_array() {
            let types: Vec<String> = type_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();
            return types.join(" | ");
        }
    }
    
    // Check for anyOf, oneOf, allOf
    if field_json.get("anyOf").is_some() {
        return "AnyOf".to_string();
    }
    if field_json.get("oneOf").is_some() {
        return "OneOf".to_string();
    }
    if field_json.get("allOf").is_some() {
        return "AllOf".to_string();
    }
    
    // Check for $ref
    if field_json.get("$ref").is_some() {
        return "Reference".to_string();
    }
    
    "Unknown".to_string()
}

/// Get a simplified field list for display purposes
pub fn get_field_names_from_schema(schema: &Schema) -> Result<Vec<String>> {
    let fields = extract_fields_from_schema(schema)?;
    Ok(fields.keys().cloned().collect())
}

/// Get field descriptions for documentation
pub fn get_field_descriptions_from_schema(schema: &Schema) -> Result<Vec<String>> {
    let fields = extract_fields_from_schema(schema)?;
    
    let descriptions: Vec<String> = fields
        .values()
        .map(|info| {
            let desc = info.description.as_deref().unwrap_or("No description");
            format!("- {}: {} ({})", info.name, desc, info.type_name)
        })
        .collect();
    
    Ok(descriptions)
}

// NEW: Functions that work with signature-filtered schemas

/// Extract fields from a signature's prompt input schema (excludes special fields)
pub fn extract_prompt_input_fields<S: Signature>() -> Result<HashMap<String, FieldInfo>> {
    let schema = S::prompt_input_schema();
    extract_fields_from_schema(&schema)
}

/// Extract fields from a signature's prompt output schema (excludes special fields)  
pub fn extract_prompt_output_fields<S: Signature>() -> Result<HashMap<String, FieldInfo>> {
    let schema = S::prompt_output_schema();
    extract_fields_from_schema(&schema)
}

/// Get field names from a signature's prompt input schema
pub fn get_prompt_input_field_names<S: Signature>() -> Result<Vec<String>> {
    let fields = extract_prompt_input_fields::<S>()?;
    Ok(fields.keys().cloned().collect())
}

/// Get field names from a signature's prompt output schema
pub fn get_prompt_output_field_names<S: Signature>() -> Result<Vec<String>> {
    let fields = extract_prompt_output_fields::<S>()?;
    Ok(fields.keys().cloned().collect())
}

/// Get field descriptions from a signature's prompt input schema
pub fn get_prompt_input_descriptions<S: Signature>() -> Result<Vec<String>> {
    let fields = extract_prompt_input_fields::<S>()?;
    
    let descriptions: Vec<String> = fields
        .values()
        .map(|info| {
            let desc = info.description.as_deref().unwrap_or("No description");
            format!("- {}: {} ({})", info.name, desc, info.type_name)
        })
        .collect();
    
    Ok(descriptions)
}

/// Get field descriptions from a signature's prompt output schema
pub fn get_prompt_output_descriptions<S: Signature>() -> Result<Vec<String>> {
    let fields = extract_prompt_output_fields::<S>()?;
    
    let descriptions: Vec<String> = fields
        .values()
        .map(|info| {
            let desc = info.description.as_deref().unwrap_or("No description");
            format!("- {}: {} ({})", info.name, desc, info.type_name)
        })
        .collect();
    
    Ok(descriptions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    
    #[derive(JsonSchema, Serialize, Deserialize)]
    struct TestStruct {
        name: String,
        age: u32,
        email: Option<String>,
    }
    
    #[test]
    fn test_extract_fields() {
        let schema = schemars::schema_for!(TestStruct);
        let fields = extract_fields_from_schema(&schema).unwrap();
        
        assert!(fields.contains_key("name"));
        assert!(fields.contains_key("age"));
        assert!(fields.contains_key("email"));
        
        assert_eq!(fields["name"].type_name, "String");
        assert_eq!(fields["age"].type_name, "Integer");
    }
}