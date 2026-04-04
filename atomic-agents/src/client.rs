pub mod types;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "google")]
pub mod google;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::Error;
use types::{ChatChunk, ChatRequest, ChatResponse};

/// Trait for LLM provider clients.
///
/// Implement this trait to add support for a new LLM provider. The framework ships
/// with implementations for OpenAI-compatible APIs, Anthropic, and Google Gemini.
///
/// All methods are async. For synchronous usage, the `AtomicAgent` provides sync
/// wrappers that use `tokio::task::block_in_place`.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Sends a chat completion request and returns the full response.
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, Error>;

    /// Sends a streaming chat completion request and returns a stream of chunks.
    async fn chat_completion_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, Error>> + Send>>, Error>;

    /// Returns the provider name for logging/debugging.
    fn provider_name(&self) -> &str;

    /// Returns the role string used for assistant messages (e.g., "assistant" or "model").
    fn assistant_role(&self) -> &str {
        "assistant"
    }

    /// Returns the role string used for tool/context injection mid-conversation.
    fn tool_result_role(&self) -> &str {
        "system"
    }
}

/// A mock LLM client for testing.
#[cfg(test)]
pub mod mock {
    use super::*;

    pub struct MockClient {
        pub response: ChatResponse,
    }

    impl MockClient {
        pub fn new(content: impl Into<String>) -> Self {
            Self {
                response: ChatResponse {
                    content: content.into(),
                    finish_reason: Some("stop".to_string()),
                    usage: None,
                },
            }
        }
    }

    #[async_trait]
    impl LlmClient for MockClient {
        async fn chat_completion(&self, _request: ChatRequest) -> Result<ChatResponse, Error> {
            Ok(self.response.clone())
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, Error>> + Send>>, Error> {
            let content = self.response.content.clone();
            let stream = futures::stream::once(async move {
                Ok(ChatChunk {
                    delta: content,
                    finish_reason: Some("stop".to_string()),
                })
            });
            Ok(Box::pin(stream))
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }
}
