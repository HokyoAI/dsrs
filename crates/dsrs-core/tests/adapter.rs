use insta::assert_snapshot;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};

use dsrs_core::{
    adapters::{
        chat_adapter::ChatAdapter,
        json_adapter::JsonAdapter,
        traits::{Adapter, AdapterConfig, Demo},
    },
    primatives::{ChatHistory, History, Signature, ToolCallSet, ToolCalls, ToolSet, Tools},
    providers::models::{AvailableTool, ContentTypes, Message, ToolCall},
};
