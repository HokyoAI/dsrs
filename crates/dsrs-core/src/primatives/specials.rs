use anyhow::Result;
use crate::providers::models::{Message, ToolCall, AvailableTool};

/// Marker trait for special fields that require custom handling in signatures
pub trait SpecialField: Send + Sync {}

/// Trait for fields that represent conversation history
/// These fields should be converted to message history instead of being formatted into prompts
pub trait History: SpecialField {
    /// Convert this history field to a sequence of messages
    fn to_messages(&self) -> Vec<Message>;
}

/// Trait for fields that define available tools for the LLM
/// These fields should be passed to the provider as tool definitions, not included in prompts
pub trait Tools: SpecialField {
    /// Convert this tools field to available tool definitions
    fn to_available_tools(&self) -> Vec<AvailableTool>;
}

/// Trait for fields that represent tool call outputs
/// These fields should be populated from tool calls returned by the LLM
pub trait ToolCalls: SpecialField {
    /// Create an instance from tool calls returned by the LLM
    fn from_tool_calls(calls: Vec<ToolCall>) -> Result<Self>
    where
        Self: Sized;
    
    /// Convert this instance back to tool calls
    fn to_tool_calls(&self) -> Vec<ToolCall>;
}

/// Example implementation for a simple history type
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatHistory {
    pub messages: Vec<Message>,
}

impl SpecialField for ChatHistory {}

impl History for ChatHistory {
    fn to_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

/// Example implementation for a simple tools type
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolSet {
    pub tools: Vec<AvailableTool>,
}

impl SpecialField for ToolSet {}

impl Tools for ToolSet {
    fn to_available_tools(&self) -> Vec<AvailableTool> {
        self.tools.clone()
    }
}

/// Example implementation for tool call outputs
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallSet {
    pub calls: Vec<ToolCall>,
}

impl SpecialField for ToolCallSet {}

impl ToolCalls for ToolCallSet {
    fn from_tool_calls(calls: Vec<ToolCall>) -> Result<Self> {
        Ok(ToolCallSet { calls })
    }
    
    fn to_tool_calls(&self) -> Vec<ToolCall> {
        self.calls.clone()
    }
}

// Convenience type aliases
pub type DefaultHistory = ChatHistory;
pub type DefaultTools = ToolSet;
pub type DefaultToolCalls = ToolCallSet;