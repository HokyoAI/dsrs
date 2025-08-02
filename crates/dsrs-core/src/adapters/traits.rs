use anyhow::{Result, anyhow};
use async_trait::async_trait;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json;

use crate::{
    primatives::Signature,
    providers::models::{ContentTypes, Message},
    providers::{CompletionConfig, CompletionProvider},
};

// Represents a demo/example for few-shot learning
#[derive(Debug, Clone, Serialize)]
pub struct Demo<I, O>
where
    I: JsonSchema + Serialize,
    O: JsonSchema + DeserializeOwned,
{
    pub inputs: I,
    pub outputs: O,
}

// Manual implementation of Deserialize to avoid conflicts
impl<'de, I, O> Deserialize<'de> for Demo<I, O>
where
    I: JsonSchema + Serialize + DeserializeOwned,
    O: JsonSchema + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DemoHelper<I, O> {
            inputs: I,
            outputs: O,
        }

        let helper = DemoHelper::deserialize(deserializer)?;
        Ok(Demo {
            inputs: helper.inputs,
            outputs: helper.outputs,
        })
    }
}

// Configuration for adapters
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    pub use_native_function_calling: bool,
    pub max_retries: usize,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            use_native_function_calling: false,
            max_retries: 3,
        }
    }
}

// Core adapter trait - generic over signature types
#[async_trait]
pub trait Adapter<S: Signature>: Send + Sync {
    fn config(&self) -> &AdapterConfig;

    // Format methods that must be implemented
    fn format_field_description(&self, schema: &Schema) -> String;
    fn format_field_structure(&self, input_schema: &Schema, output_schema: &Schema) -> String;
    fn format_task_description(&self, instructions: &str) -> String;
    fn format_user_message_content(&self, inputs: &S::Inputs, schema: &Schema) -> String;
    fn format_assistant_message_content(&self, outputs: &S::Outputs, schema: &Schema) -> String;

    // Parse the completion back to the output type
    fn parse(&self, completion: &str, schema: &Schema) -> Result<S::Outputs>;

    // Core functionality with default implementations
    async fn generate(
        &self,
        provider: &impl CompletionProvider,
        base_config: CompletionConfig,
        signature: &S,
        instructions: &str,
        demos: &[Demo<S::Inputs, S::Outputs>],
        inputs: &S::Inputs,
    ) -> Result<S::Outputs> {
        // Extract special fields from inputs
        let history = signature.extract_history(inputs);
        let tools = signature.extract_tools(inputs);

        // Filter inputs to only contain prompt-relevant fields
        let filtered_inputs = signature.filter_special_fields(inputs);

        // Get filtered schemas for prompt formatting
        let input_schema = S::prompt_input_schema();
        let output_schema = S::prompt_output_schema();

        // Format messages using filtered inputs and schemas
        let mut messages = self.format_messages_filtered(
            signature,
            instructions,
            demos,
            &filtered_inputs,
            &input_schema,
            &output_schema,
        )?;

        // Prepend history if present (insert after system message)
        if let Some(hist) = history {
            if !messages.is_empty() {
                messages.splice(1..1, hist);
            } else {
                messages.extend(hist);
            }
        }

        // Build enhanced config with tools
        let config = CompletionConfig {
            model: base_config.model,
            tools: tools.or(base_config.tools),
        };

        let all_messages = std::sync::Arc::new(tokio::sync::RwLock::new(messages));

        // Try with retries
        for attempt in 0..self.config().max_retries {
            match provider
                .complete(all_messages.clone(), config.clone())
                .await
            {
                Ok(response) => {
                    if let Message::Assistant {
                        content: Some(ContentTypes::Text(text)),
                        tool_calls,
                    } = response
                    {
                        // Parse regular outputs
                        match self.parse(&text, &output_schema) {
                            Ok(mut outputs) => {
                                // Handle tool calls if present
                                if let Some(calls) = tool_calls {
                                    signature.inject_tool_calls(&mut outputs, calls.clone())?;
                                    // Use signature's merge function for final result
                                    return signature.merge_special_outputs(outputs, Some(calls));
                                } else {
                                    return signature.merge_special_outputs(outputs, None);
                                }
                            }
                            Err(e) if attempt < self.config().max_retries - 1 => {
                                eprintln!("Parse error on attempt {}: {}", attempt + 1, e);
                                continue;
                            }
                            Err(e) => return Err(e),
                        }
                    } else if let Message::Assistant {
                        content: None,
                        tool_calls: Some(calls),
                    } = response
                    {
                        // Handle tool-only responses
                        let mut outputs = serde_json::from_value(serde_json::json!({}))?;
                        signature.inject_tool_calls(&mut outputs, calls.clone())?;
                        return signature.merge_special_outputs(outputs, Some(calls));
                    } else {
                        return Err(anyhow!(
                            "Expected assistant message with text content or tool calls"
                        ));
                    }
                }
                Err(e) if attempt < self.config().max_retries - 1 => {
                    eprintln!("Provider error on attempt {}: {}", attempt + 1, e);
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }

        Err(anyhow!(
            "Failed after {} attempts",
            self.config().max_retries
        ))
    }

    // Original format_messages for backward compatibility
    fn format_messages(
        &self,
        instructions: &str,
        demos: &[Demo<S::Inputs, S::Outputs>],
        inputs: &S::Inputs,
    ) -> Result<Vec<Message>> {
        // Use full schemas for backward compatibility
        let input_schema = self.get_input_schema();
        let output_schema = self.get_output_schema();

        self.format_messages_with_schemas(
            instructions,
            demos,
            inputs,
            &input_schema,
            &output_schema,
        )
    }

    // New format_messages_filtered that uses signature-provided schemas
    fn format_messages_filtered(
        &self,
        _signature: &S,
        instructions: &str,
        demos: &[Demo<S::Inputs, S::Outputs>],
        inputs: &S::Inputs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> Result<Vec<Message>> {
        self.format_messages_with_schemas(instructions, demos, inputs, input_schema, output_schema)
    }

    // Common implementation for both message formatting approaches
    fn format_messages_with_schemas(
        &self,
        instructions: &str,
        demos: &[Demo<S::Inputs, S::Outputs>],
        inputs: &S::Inputs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> Result<Vec<Message>> {
        let mut messages = Vec::new();

        // System message
        let system_content = format!(
            "{}\n{}\n{}",
            self.format_field_description(input_schema),
            self.format_field_structure(input_schema, output_schema),
            self.format_task_description(instructions)
        );
        messages.push(Message::system(system_content));

        // Add few-shot examples
        messages.extend(self.format_demos_with_schemas(demos, input_schema, output_schema)?);

        // Add current input
        let user_content = self.format_user_message_content(inputs, input_schema);
        messages.push(Message::user(user_content));

        Ok(messages)
    }

    fn format_demos(&self, demos: &[Demo<S::Inputs, S::Outputs>]) -> Result<Vec<Message>> {
        let input_schema = self.get_input_schema();
        let output_schema = self.get_output_schema();
        self.format_demos_with_schemas(demos, &input_schema, &output_schema)
    }

    fn format_demos_with_schemas(
        &self,
        demos: &[Demo<S::Inputs, S::Outputs>],
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> Result<Vec<Message>> {
        let mut messages = Vec::new();

        for demo in demos {
            messages.push(Message::user(
                self.format_user_message_content(&demo.inputs, input_schema),
            ));
            messages.push(Message::assistant(
                Some(self.format_assistant_message_content(&demo.outputs, output_schema)),
                None,
            ));
        }

        Ok(messages)
    }

    // Helper methods to get schemas
    fn get_input_schema(&self) -> Schema {
        schemars::schema_for!(S::Inputs)
    }

    fn get_output_schema(&self) -> Schema {
        schemars::schema_for!(S::Outputs)
    }
}
