use std::future::Future;

use super::{CompletionConfig, Message, ProviderError};

use std::sync::Arc;
use tokio::sync::RwLock;

pub trait CompletionProvider: Send + Sync {
    fn complete(
        &self,
        messages: Arc<RwLock<Vec<Message>>>,
        config: CompletionConfig,
    ) -> impl Future<Output = Result<Message, ProviderError>> + Send;
}
