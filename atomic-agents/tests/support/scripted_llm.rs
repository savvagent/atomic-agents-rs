//! A scripted [`LlmClient`] that answers each request with the next canned
//! JSON value and records every request it receives.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use atomic_agents::client::LlmClient;
use atomic_agents::client::types::{ChatChunk, ChatRequest, ChatResponse};
use atomic_agents::error::Error;
use futures::Stream;

/// Replays `responses` in order. In `ExtractionMode::Tools` the agent reads
/// the response content as the tool-call arguments, so each scripted value is
/// exactly what the model would pass to the output function.
pub struct ScriptedLlm {
    responses: Mutex<VecDeque<serde_json::Value>>,
    requests: Arc<Mutex<Vec<ChatRequest>>>,
}

impl ScriptedLlm {
    /// Returns the client and a handle to the requests it will record.
    pub fn new(
        responses: impl IntoIterator<Item = serde_json::Value>,
    ) -> (Self, Arc<Mutex<Vec<ChatRequest>>>) {
        let requests: Arc<Mutex<Vec<ChatRequest>>> = Arc::default();
        let llm = Self {
            responses: Mutex::new(responses.into_iter().collect()),
            requests: requests.clone(),
        };
        (llm, requests)
    }
}

#[async_trait]
impl LlmClient for ScriptedLlm {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        self.requests.lock().unwrap().push(request);
        let next = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| Error::Config("scripted responses exhausted".to_string()))?;
        Ok(ChatResponse {
            content: next.to_string(),
            finish_reason: Some("stop".to_string()),
            usage: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, Error>> + Send>>, Error> {
        Err(Error::Config("ScriptedLlm does not stream".to_string()))
    }

    fn provider_name(&self) -> &str {
        "scripted"
    }
}
