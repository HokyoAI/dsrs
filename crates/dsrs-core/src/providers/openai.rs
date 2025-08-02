use super::CompletionProvider;
use super::ProviderError;
use super::models::*;

use async_openai::types::{
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessageContent,
};
use async_openai::{Client, config::OpenAIConfig};

use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
    ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType,
    CreateChatCompletionRequestArgs, FunctionCall, FunctionObjectArgs, ServiceTier,
};

use std::sync::Arc;
use tokio::sync::RwLock;

pub struct OpenAIProvider {
    client: Client<OpenAIConfig>,
}

impl OpenAIProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key);
        let config = if let Some(url) = base_url {
            config.with_api_base(url)
        } else {
            config
        };
        let client = Client::with_config(config);
        OpenAIProvider { client }
    }
}

impl From<&ContentTypes> for ChatCompletionRequestUserMessageContent {
    fn from(content: &ContentTypes) -> Self {
        match content {
            ContentTypes::Text(text) => ChatCompletionRequestUserMessageContent::Text(text.clone()),
        }
    }
}

impl From<&ContentTypes> for ChatCompletionRequestAssistantMessageContent {
    fn from(content: &ContentTypes) -> Self {
        match content {
            ContentTypes::Text(text) => {
                ChatCompletionRequestAssistantMessageContent::Text(text.clone())
            }
        }
    }
}

impl From<&ContentTypes> for ChatCompletionRequestToolMessageContent {
    fn from(content: &ContentTypes) -> Self {
        match content {
            ContentTypes::Text(text) => ChatCompletionRequestToolMessageContent::Text(text.clone()),
        }
    }
}

impl From<&ContentTypes> for ChatCompletionRequestSystemMessageContent {
    fn from(content: &ContentTypes) -> Self {
        match content {
            ContentTypes::Text(text) => {
                ChatCompletionRequestSystemMessageContent::Text(text.clone())
            }
        }
    }
}

impl From<&ToolCall> for ChatCompletionMessageToolCall {
    fn from(tool_call: &ToolCall) -> Self {
        ChatCompletionMessageToolCall {
            id: tool_call.id.clone(),
            r#type: async_openai::types::ChatCompletionToolType::Function,
            function: FunctionCall {
                name: tool_call.name.clone(),
                arguments: serde_json::to_string(&tool_call.arguments).unwrap(),
            },
        }
    }
}

impl From<&Message> for ChatCompletionRequestMessage {
    fn from(message: &Message) -> Self {
        match message {
            Message::User { content } => ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(content)
                    .build()
                    .unwrap(),
            ),
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
                if let Some(calls) = tool_calls {
                    if !calls.is_empty() {
                        let openai_tool_calls: Vec<ChatCompletionMessageToolCall> = calls
                            .iter()
                            .map(ChatCompletionMessageToolCall::from)
                            .collect();
                        builder.tool_calls(openai_tool_calls);
                    }
                }
                if let Some(content) = content {
                    builder.content(content);
                }
                let message = builder.build().unwrap();
                ChatCompletionRequestMessage::Assistant(message)
            }
            Message::System { content } => ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(content)
                    .build()
                    .unwrap(),
            ),
            Message::Tool {
                content,
                tool_call_id,
            } => ChatCompletionRequestMessage::Tool(
                ChatCompletionRequestToolMessageArgs::default()
                    .content(content)
                    .tool_call_id(tool_call_id)
                    .build()
                    .unwrap(),
            ),
        }
    }
}

impl From<&AvailableTool> for ChatCompletionTool {
    fn from(tool: &AvailableTool) -> Self {
        let mut builder = FunctionObjectArgs::default();

        builder
            .name(tool.name.clone())
            .description(tool.desc.clone())
            .strict(true);

        if let Some(schema) = tool.input_schema_json.clone() {
            builder.parameters(schema);
        }

        let func = builder.build().unwrap();

        ChatCompletionToolArgs::default()
            .r#type(ChatCompletionToolType::Function)
            .function(func)
            .build()
            .unwrap()
    }
}

impl From<ChatCompletionMessageToolCall> for ToolCall {
    fn from(tool_call: ChatCompletionMessageToolCall) -> Self {
        ToolCall {
            id: tool_call.id,
            name: tool_call.function.name,
            arguments: serde_json::to_value(tool_call.function.arguments).unwrap(),
        }
    }
}

impl CompletionProvider for OpenAIProvider {
    async fn complete(
        &self,
        messages: Arc<RwLock<Vec<Message>>>,
        config: CompletionConfig,
    ) -> Result<Message, ProviderError> {
        // Clone the messages and immediately release the lock
        let request_messages = {
            let guard = messages.read().await;
            guard
                .iter()
                .map(ChatCompletionRequestMessage::from)
                .collect::<Vec<ChatCompletionRequestMessage>>()
        };

        let available_tools = match config.tools {
            Some(tools) => {
                let tool_vec = tools
                    .iter()
                    .map(ChatCompletionTool::from)
                    .collect::<Vec<ChatCompletionTool>>();
                Some(tool_vec)
            }
            None => None,
        };

        let mut builder = CreateChatCompletionRequestArgs::default();

        let request = if let Some(tools) = available_tools {
            builder
                .messages(request_messages)
                .model(config.model)
                .tools(tools)
                .service_tier(ServiceTier::Flex) // Groq sending unsupported service tier back, need to specify
                .build()?
        } else {
            builder
                .messages(request_messages)
                .model(config.model)
                .service_tier(ServiceTier::Flex) // Groq sending unsupported service tier back, need to specify
                .build()?
        };

        let response = self.client.chat().create(request).await?;
        let first_choice = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| {
                let content = choice.message.content;
                let calls = choice
                    .message
                    .tool_calls
                    .map(|calls| calls.into_iter().map(ToolCall::from).collect());
                Some((content, calls))
            })
            .unwrap();

        Ok(Message::assistant(first_choice.0, first_choice.1))
    }
}
