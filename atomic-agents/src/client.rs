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

    /// Returns the extraction mode agents use with this client when their
    /// `AgentConfig` does not set one explicitly.
    ///
    /// Defaults to [`ExtractionMode::Tools`](crate::agent::ExtractionMode::Tools).
    fn default_extraction_mode(&self) -> crate::agent::ExtractionMode {
        crate::agent::ExtractionMode::Tools
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
    use std::sync::Mutex;

    pub struct MockClient {
        pub response: ChatResponse,
        /// Chunks emitted by `chat_completion_stream` (defaults to the response content).
        pub stream_chunks: Vec<String>,
        /// When set, `chat_completion` and `chat_completion_stream` fail.
        pub fail: bool,
        /// When set, the stream yields an error after its chunks.
        pub stream_error: bool,
        pub mode: crate::agent::ExtractionMode,
        /// Every request received, in order.
        pub requests: Mutex<Vec<ChatRequest>>,
    }

    impl MockClient {
        pub fn new(content: impl Into<String>) -> Self {
            let content = content.into();
            Self {
                stream_chunks: vec![content.clone()],
                response: ChatResponse {
                    content,
                    finish_reason: Some("stop".to_string()),
                    usage: None,
                },
                fail: false,
                stream_error: false,
                mode: crate::agent::ExtractionMode::Tools,
                requests: Mutex::new(Vec::new()),
            }
        }

        pub fn failing() -> Self {
            let mut client = Self::new("");
            client.fail = true;
            client
        }

        pub fn with_stream_chunks(mut self, chunks: &[&str]) -> Self {
            self.stream_chunks = chunks.iter().map(|c| c.to_string()).collect();
            self
        }

        pub fn with_mode(mut self, mode: crate::agent::ExtractionMode) -> Self {
            self.mode = mode;
            self
        }
    }

    #[async_trait]
    impl LlmClient for MockClient {
        async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
            self.requests.lock().unwrap().push(request);
            if self.fail {
                return Err(Error::Config("mock failure".to_string()));
            }
            Ok(self.response.clone())
        }

        async fn chat_completion_stream(
            &self,
            request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, Error>> + Send>>, Error> {
            self.requests.lock().unwrap().push(request);
            if self.fail {
                return Err(Error::Config("mock failure".to_string()));
            }
            let mut items: Vec<Result<ChatChunk, Error>> = self
                .stream_chunks
                .iter()
                .map(|delta| {
                    Ok(ChatChunk {
                        delta: delta.clone(),
                        finish_reason: None,
                    })
                })
                .collect();
            if self.stream_error {
                items.push(Err(Error::Config("mock stream failure".to_string())));
            }
            Ok(Box::pin(futures::stream::iter(items)))
        }

        fn provider_name(&self) -> &str {
            "mock"
        }

        fn default_extraction_mode(&self) -> crate::agent::ExtractionMode {
            self.mode
        }
    }
}
