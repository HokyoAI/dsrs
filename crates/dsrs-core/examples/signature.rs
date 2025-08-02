use anyhow::Result;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};

use dsrs_core::{
    primatives::{ChatHistory, History, Signature, ToolCallSet, ToolCalls, ToolSet, Tools},
    providers::models::{AvailableTool, Message, ToolCall},
};

/// Example inputs that include both regular fields and special fields
#[derive(Serialize, Deserialize, JsonSchema, Clone)]
pub struct EnhancedInputs {
    // Regular fields that will appear in prompts
    pub query: String,
    pub context: String,

    // Special fields that get handled differently
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)] // Exclude from JsonSchema so it doesn't appear in prompts
    pub history: Option<ChatHistory>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)] // Exclude from JsonSchema so it doesn't appear in prompts
    pub tools: Option<ToolSet>,
}

/// Example outputs that can include tool calls
#[derive(Serialize, Deserialize, JsonSchema)]
pub struct EnhancedOutputs {
    // Regular fields
    pub answer: String,
    pub confidence: f64,

    // Special field for tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)] // Exclude from JsonSchema so it doesn't appear in prompts
    pub tool_calls: Option<ToolCallSet>,
}

/// Alternative approach: separate prompt types (if you prefer more explicit control)
#[derive(Serialize, Deserialize, JsonSchema, Clone)]
pub struct PromptInputs {
    pub query: String,
    pub context: String,
    // Only the fields we want in prompts
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct PromptOutputs {
    pub answer: String,
    pub confidence: f64,
    // Only the fields we want in prompts
}

/// Example signature implementation using the schemars(skip) approach
pub struct EnhancedSignature {
    instructions: String,
}

impl EnhancedSignature {
    pub fn new() -> Self {
        Self {
            instructions: "Answer the user's question based on the provided context.".to_string(),
        }
    }
}

impl Signature for EnhancedSignature {
    type Inputs = EnhancedInputs;
    type Outputs = EnhancedOutputs;

    fn set_instructions(&mut self, instructions: String) {
        self.instructions = instructions;
    }

    fn get_instructions(&self) -> &str {
        &self.instructions
    }

    fn name(&self) -> &str {
        "EnhancedQA"
    }

    fn desc(&self) -> &str {
        "Enhanced question answering with history and tool support"
    }

    // Extract special fields from inputs
    fn extract_history(&self, inputs: &Self::Inputs) -> Option<Vec<Message>> {
        inputs.history.as_ref().map(|h| h.to_messages())
    }

    fn extract_tools(&self, inputs: &Self::Inputs) -> Option<Vec<AvailableTool>> {
        inputs.tools.as_ref().map(|t| t.to_available_tools())
    }

    // Inject tool calls into outputs
    fn inject_tool_calls(&self, outputs: &mut Self::Outputs, calls: Vec<ToolCall>) -> Result<()> {
        outputs.tool_calls = Some(ToolCallSet::from_tool_calls(calls)?);
        Ok(())
    }

    // Filter special fields for prompt generation
    fn filter_special_fields(&self, inputs: &Self::Inputs) -> Self::Inputs {
        Self::Inputs {
            query: inputs.query.clone(),
            context: inputs.context.clone(),
            history: None, // Filter out special fields
            tools: None,   // Filter out special fields
        }
    }

    // Merge special outputs (in this case, just return the outputs as-is since tool_calls is already injected)
    fn merge_special_outputs(
        &self,
        regular: Self::Outputs,
        _calls: Option<Vec<ToolCall>>,
    ) -> Result<Self::Outputs> {
        Ok(regular)
    }
}

/// Alternative signature implementation using separate prompt types
pub struct ExplicitPromptSignature {
    instructions: String,
}

impl ExplicitPromptSignature {
    pub fn new() -> Self {
        Self {
            instructions: "Answer the user's question based on the provided context.".to_string(),
        }
    }

    // Convert full inputs to prompt inputs
    fn to_prompt_inputs(&self, inputs: &EnhancedInputs) -> PromptInputs {
        PromptInputs {
            query: inputs.query.clone(),
            context: inputs.context.clone(),
        }
    }

    // Convert prompt outputs to full outputs
    fn from_prompt_outputs(
        &self,
        prompt_outputs: PromptOutputs,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Result<EnhancedOutputs> {
        Ok(EnhancedOutputs {
            answer: prompt_outputs.answer,
            confidence: prompt_outputs.confidence,
            tool_calls: if let Some(calls) = tool_calls {
                Some(ToolCallSet::from_tool_calls(calls)?)
            } else {
                None
            },
        })
    }
}

impl Signature for ExplicitPromptSignature {
    type Inputs = EnhancedInputs;
    type Outputs = EnhancedOutputs;

    fn set_instructions(&mut self, instructions: String) {
        self.instructions = instructions;
    }

    fn get_instructions(&self) -> &str {
        &self.instructions
    }

    fn name(&self) -> &str {
        "ExplicitPromptQA"
    }

    fn desc(&self) -> &str {
        "Question answering with explicit prompt type separation"
    }

    // Override schema methods to use prompt types
    fn prompt_input_schema() -> Schema {
        schemars::schema_for!(PromptInputs)
    }

    fn prompt_output_schema() -> Schema {
        schemars::schema_for!(PromptOutputs)
    }

    // Extract special fields
    fn extract_history(&self, inputs: &Self::Inputs) -> Option<Vec<Message>> {
        inputs.history.as_ref().map(|h| h.to_messages())
    }

    fn extract_tools(&self, inputs: &Self::Inputs) -> Option<Vec<AvailableTool>> {
        inputs.tools.as_ref().map(|t| t.to_available_tools())
    }

    // Filter special fields by converting to prompt inputs
    fn filter_special_fields(&self, inputs: &Self::Inputs) -> Self::Inputs {
        let prompt_inputs = self.to_prompt_inputs(inputs);
        Self::Inputs {
            query: prompt_inputs.query,
            context: prompt_inputs.context,
            history: None,
            tools: None,
        }
    }

    // Handle tool call injection and merging
    fn inject_tool_calls(&self, _outputs: &mut Self::Outputs, _calls: Vec<ToolCall>) -> Result<()> {
        // Tool calls are handled in merge_special_outputs for this approach
        Ok(())
    }

    fn merge_special_outputs(
        &self,
        regular: Self::Outputs,
        calls: Option<Vec<ToolCall>>,
    ) -> Result<Self::Outputs> {
        // Convert to prompt outputs, then back to full outputs with tool calls
        let prompt_outputs = PromptOutputs {
            answer: regular.answer,
            confidence: regular.confidence,
        };
        self.from_prompt_outputs(prompt_outputs, calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_signature_schema_filtering() {
        // Test that the filtered schema doesn't include special fields
        let schema = EnhancedSignature::prompt_input_schema();
        let schema_json = serde_json::to_value(&schema).unwrap();

        // The schema should not contain 'history' or 'tools' fields due to schemars(skip)
        if let Some(properties) = schema_json.get("object").and_then(|o| o.get("properties")) {
            assert!(!properties.as_object().unwrap().contains_key("history"));
            assert!(!properties.as_object().unwrap().contains_key("tools"));
            assert!(properties.as_object().unwrap().contains_key("query"));
            assert!(properties.as_object().unwrap().contains_key("context"));
        }
    }

    #[test]
    fn test_explicit_prompt_signature_schemas() {
        // Test that the explicit prompt schemas work correctly
        let input_schema = ExplicitPromptSignature::prompt_input_schema();
        let output_schema = ExplicitPromptSignature::prompt_output_schema();

        // These should be the schemas for PromptInputs and PromptOutputs respectively
        let input_json = serde_json::to_value(&input_schema).unwrap();
        let output_json = serde_json::to_value(&output_schema).unwrap();

        // Input schema should only have query and context
        if let Some(input_props) = input_json.get("object").and_then(|o| o.get("properties")) {
            let props = input_props.as_object().unwrap();
            assert_eq!(props.len(), 2);
            assert!(props.contains_key("query"));
            assert!(props.contains_key("context"));
        }

        // Output schema should only have answer and confidence
        if let Some(output_props) = output_json.get("object").and_then(|o| o.get("properties")) {
            let props = output_props.as_object().unwrap();
            assert_eq!(props.len(), 2);
            assert!(props.contains_key("answer"));
            assert!(props.contains_key("confidence"));
        }
    }
}
