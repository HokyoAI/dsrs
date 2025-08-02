pub use schemars::{JsonSchema, Schema};
pub use serde::{Deserialize, Serialize};

// MARK: Base

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContentTypes {
    Text(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    System {
        content: ContentTypes,
    },
    User {
        content: ContentTypes,
    },
    Assistant {
        content: Option<ContentTypes>,
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        content: ContentTypes,
        tool_call_id: String,
    },
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message::User {
            content: ContentTypes::Text(content.into()),
        }
    }

    pub fn assistant(
        content: Option<impl Into<String>>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        Message::Assistant {
            content: content.map(|c| ContentTypes::Text(c.into())),
            tool_calls,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: ContentTypes::Text(content.into()),
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Message::Tool {
            content: ContentTypes::Text(content.into()),
            tool_call_id: tool_call_id.into(),
        }
    }
}

// MARK: Completions

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AvailableTool {
    pub name: String,
    pub desc: String,
    pub input_schema_json: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionConfig {
    pub model: String,
    pub tools: Option<Vec<AvailableTool>>,
}
