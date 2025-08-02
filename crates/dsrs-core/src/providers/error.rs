use async_openai::error::OpenAIError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("OpenAI error occurred: {0}")]
    OpenAIError(#[from] OpenAIError),
}
