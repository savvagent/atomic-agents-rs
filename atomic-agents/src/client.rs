pub mod types;

#[cfg(feature = "openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai")))]
pub mod openai;

#[cfg(feature = "anthropic")]
#[cfg_attr(docsrs, doc(cfg(feature = "anthropic")))]
pub mod anthropic;

#[cfg(feature = "google")]
#[cfg_attr(docsrs, doc(cfg(feature = "google")))]
pub mod google;

/// Whether an extra-parameter key would override the tools or the tool
/// choice a provider client sets. Those clients let the model choose (a
/// forced choice is rejected by current models) and send exactly the tools
/// the request names, so `extra` may not change either; such keys are
/// skipped with a warning.
#[cfg(any(feature = "anthropic", feature = "google"))]
pub(crate) fn is_tool_choice_key(key: &str) -> bool {
    matches!(key, "tool_choice" | "toolConfig" | "tool_config" | "tools")
}

/// Test doubles shared by the provider clients' stream tests.
#[cfg(all(test, any(feature = "anthropic", feature = "google")))]
pub(crate) mod test_support {
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use bytes::Bytes;
    use futures::Stream;

    /// Yields its chunks, then `None` once, and panics if polled again: a
    /// stream that is not fused may do anything after it ends.
    pub(crate) struct PanicAfterEnd {
        chunks: VecDeque<Bytes>,
        ended: bool,
    }

    impl PanicAfterEnd {
        pub(crate) fn new(chunks: Vec<Bytes>) -> Self {
            Self {
                chunks: chunks.into(),
                ended: false,
            }
        }
    }

    impl Stream for PanicAfterEnd {
        type Item = Result<Bytes, reqwest::Error>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            assert!(!this.ended, "polled after it returned None");
            match this.chunks.pop_front() {
                Some(b) => Poll::Ready(Some(Ok(b))),
                None => {
                    this.ended = true;
                    Poll::Ready(None)
                }
            }
        }
    }
}

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
/// All methods are async. For synchronous usage, the `AtomicAgent` provides a sync
/// wrapper that uses `tokio::task::block_in_place` with the `native` feature and
/// `futures::executor::block_on` without it (host calls are synchronous in a guest).
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
