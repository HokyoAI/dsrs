use anyhow::Result;
use schemars::Schema;
use crate::providers::models::{Message, ToolCall, AvailableTool};

pub trait Signature: Send + Sync {
    type Inputs: schemars::JsonSchema + serde::Serialize + Send + Sync + Clone;
    type Outputs: schemars::JsonSchema + serde::de::DeserializeOwned + serde::Serialize + Send + Sync;

    fn set_instructions(&mut self, instructions: String);
    fn get_instructions(&self) -> &str;

    fn name(&self) -> &str;
    fn desc(&self) -> &str;

    // Schema filtering methods - return schemas excluding special fields
    fn prompt_input_schema() -> Schema {
        // Default: use full input schema (for backward compatibility)
        schemars::schema_for!(Self::Inputs)
    }
    
    fn prompt_output_schema() -> Schema {
        // Default: use full output schema (for backward compatibility)
        schemars::schema_for!(Self::Outputs)
    }

    // Special field extraction methods - default implementations return None
    fn extract_history(&self, _inputs: &Self::Inputs) -> Option<Vec<Message>> {
        None
    }
    
    fn extract_tools(&self, _inputs: &Self::Inputs) -> Option<Vec<AvailableTool>> {
        None
    }
    
    // Special field injection for outputs
    fn inject_tool_calls(&self, _outputs: &mut Self::Outputs, _calls: Vec<ToolCall>) -> Result<()> {
        // Default: do nothing (for signatures that don't handle tool calls)
        Ok(())
    }
    
    // Create a copy of inputs suitable for prompt generation (with special fields filtered out)
    // Default implementation returns the original inputs (for backward compatibility)
    fn filter_special_fields(&self, inputs: &Self::Inputs) -> Self::Inputs {
        inputs.clone()
    }
    
    // Merge regular outputs with tool call results
    // Default implementation returns the regular outputs unchanged
    fn merge_special_outputs(&self, regular: Self::Outputs, _calls: Option<Vec<ToolCall>>) -> Result<Self::Outputs> {
        Ok(regular)
    }
}
