use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::Stream;
use reqwest::Client;

use super::types::{ChatChunk, ChatRequest, ChatResponse, Usage};
use super::LlmClient;
use crate::error::{ClientError, Error};

/// Client for the Anthropic Messages API.
///
/// Implements the Anthropic-specific message format with content blocks,
/// system prompt as a top-level parameter, and tool use via content blocks.
pub struct AnthropicClient {
    client: Client,
    api_key: String,
    base_url: String,
    default_max_tokens: u32,
}

impl AnthropicClient {
    /// Creates a new Anthropic client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            default_max_tokens: 4096,
        }
    }

    /// Sets the default max_tokens for requests.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Creates a client with a custom base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Creates a client with a custom reqwest client.
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    fn build_request_body(&self, request: &ChatRequest) -> serde_json::Value {
        // Anthropic uses system as a top-level parameter, not a message
        let mut system_text = String::new();
        let mut messages = Vec::new();

        for msg in &request.messages {
            if msg.role == "system" {
                // Accumulate system messages into top-level system param
                if let Some(text) = msg.content.as_str() {
                    if !system_text.is_empty() {
                        system_text.push('\n');
                    }
                    system_text.push_str(text);
                }
            } else {
                messages.push(serde_json::json!({
                    "role": msg.role,
                    "content": msg.content,
                }));
            }
        }

        let max_tokens = request
            .extra
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_max_tokens as u64);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        });

        if !system_text.is_empty() {
            body["system"] = serde_json::json!(system_text);
        }

        if request.stream {
            body["stream"] = serde_json::json!(true);
        }

        // Convert tools to Anthropic format
        if let Some(tools) = &request.tools {
            let anthropic_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.function.name,
                        "description": t.function.description,
                        "input_schema": t.function.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(anthropic_tools);
            body["tool_choice"] = serde_json::json!({"type": "any"});
        }

        // Merge extra parameters (skip max_tokens since we handled it)
        if let serde_json::Value::Object(map) = &mut body {
            for (k, v) in &request.extra {
                if k != "max_tokens" {
                    map.insert(k.clone(), v.clone());
                }
            }
        }

        body
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        let body = self.build_request_body(&request);

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ClientError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            if status.as_u16() == 401 {
                return Err(Error::Client(ClientError::AuthFailed));
            }
            if status.as_u16() == 429 {
                return Err(Error::Client(ClientError::RateLimited {
                    retry_after: None,
                }));
            }
            return Err(Error::Client(ClientError::Api {
                status: status.as_u16(),
                message: error_body,
            }));
        }

        let response_json: serde_json::Value =
            response.json().await.map_err(ClientError::Http)?;

        // Extract content from Anthropic response
        // Anthropic returns content as an array of content blocks
        let content = if let Some(content_blocks) = response_json.get("content") {
            let mut result = String::new();
            if let Some(blocks) = content_blocks.as_array() {
                for block in blocks {
                    match block.get("type").and_then(|t| t.as_str()) {
                        Some("tool_use") => {
                            // Tool use block contains structured output
                            if let Some(input) = block.get("input") {
                                result = serde_json::to_string(input).unwrap_or_default();
                            }
                        }
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                result = text.to_string();
                            }
                        }
                        _ => {}
                    }
                }
            }
            result
        } else {
            String::new()
        };

        let usage = response_json.get("usage").map(|u| Usage {
            prompt_tokens: u
                .get("input_tokens")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
            completion_tokens: u
                .get("output_tokens")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
            total_tokens: None, // Anthropic doesn't return total
        });

        let finish_reason = response_json
            .get("stop_reason")
            .and_then(|f| f.as_str())
            .map(String::from);

        Ok(ChatResponse {
            content,
            finish_reason,
            usage,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, Error>> + Send>>, Error> {
        let mut req = request.clone();
        req.stream = true;
        let body = self.build_request_body(&req);

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ClientError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(Error::Client(ClientError::Api {
                status: status.as_u16(),
                message: error_body,
            }));
        }

        let byte_stream = response.bytes_stream();
        let stream = AnthropicSseStream::new(byte_stream);
        Ok(Box::pin(stream))
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

/// Parses Anthropic's SSE stream format into ChatChunks.
struct AnthropicSseStream<S> {
    inner: S,
    buffer: String,
}

impl<S> AnthropicSseStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }
}

impl<S> Stream for AnthropicSseStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send,
{
    type Item = Result<ChatChunk, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if let Some(chunk) = parse_anthropic_sse(&mut this.buffer) {
                return std::task::Poll::Ready(Some(Ok(chunk)));
            }

            let inner = Pin::new(&mut this.inner);
            match inner.poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(Error::Client(
                        ClientError::Http(e),
                    ))));
                }
                std::task::Poll::Ready(None) => {
                    if let Some(chunk) = parse_anthropic_sse(&mut this.buffer) {
                        return std::task::Poll::Ready(Some(Ok(chunk)));
                    }
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

fn parse_anthropic_sse(buffer: &mut String) -> Option<ChatChunk> {
    loop {
        let newline_pos = buffer.find("\n\n")?;
        let event_str = buffer[..newline_pos].to_string();
        *buffer = buffer[newline_pos + 2..].to_string();

        let mut event_type = "";
        let mut data_str = "";

        for line in event_str.lines() {
            if let Some(et) = line.strip_prefix("event: ") {
                event_type = et.trim();
            } else if let Some(d) = line.strip_prefix("data: ") {
                data_str = d.trim();
            }
        }

        // We need to own the strings before parsing
        let event_type = event_type.to_string();
        let data_str = data_str.to_string();

        match event_type.as_str() {
            "content_block_delta" => {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data_str) {
                    let delta = json.get("delta").unwrap_or(&serde_json::Value::Null);
                    let content = match delta.get("type").and_then(|t| t.as_str()) {
                        Some("text_delta") => delta
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string(),
                        Some("input_json_delta") => delta
                            .get("partial_json")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string(),
                        _ => String::new(),
                    };
                    if !content.is_empty() {
                        return Some(ChatChunk {
                            delta: content,
                            finish_reason: None,
                        });
                    }
                }
            }
            "message_delta" => {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data_str) {
                    let stop_reason = json
                        .get("delta")
                        .and_then(|d| d.get("stop_reason"))
                        .and_then(|s| s.as_str())
                        .map(String::from);
                    if stop_reason.is_some() {
                        return Some(ChatChunk {
                            delta: String::new(),
                            finish_reason: stop_reason,
                        });
                    }
                }
            }
            "message_stop" => {
                return Some(ChatChunk {
                    delta: String::new(),
                    finish_reason: Some("end_turn".to_string()),
                });
            }
            _ => {
                // Skip other event types (message_start, content_block_start, etc.)
            }
        }
    }
}
