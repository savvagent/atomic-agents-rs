use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::Stream;
use reqwest::Client;

use super::types::{ChatChunk, ChatRequest, ChatResponse, Usage};
use super::{LlmClient, is_tool_choice_key};
use crate::error::{ClientError, Error};

/// Client for OpenAI-compatible chat completion APIs.
///
/// Works with OpenAI, Azure OpenAI, Groq, Together, Ollama, OpenRouter,
/// and any other API that implements the OpenAI chat completions format.
pub struct OpenAiClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAiClient {
    /// Creates a new OpenAI client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    /// Creates a client with a custom base URL (for compatible providers).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Creates a client with a custom reqwest client (for proxy, timeouts, etc.).
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    fn build_request_body(&self, request: &ChatRequest) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": request.model,
            "messages": request.messages.iter().map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            }).collect::<Vec<_>>(),
        });

        if request.stream {
            body["stream"] = serde_json::json!(true);
        }

        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::to_value(tools).unwrap();
        }

        if let Some(fmt) = &request.response_format {
            body["response_format"] = serde_json::to_value(fmt).unwrap();
        }

        // Merge extra parameters. Keys that choose a tool are skipped, as in
        // the other clients: the request's tools are what the agent sends,
        // and a forced choice is rejected by current models.
        if let serde_json::Value::Object(map) = &mut body {
            for (k, v) in &request.extra {
                if is_tool_choice_key(k) {
                    tracing::warn!(key = %k, "extra parameter may not override tool choice; skipped");
                    continue;
                }
                map.insert(k.clone(), v.clone());
            }
        }

        body
    }

    fn extract_content(response_json: &serde_json::Value) -> Result<String, Error> {
        let choice = response_json
            .get("choices")
            .and_then(|c| c.get(0))
            .ok_or_else(|| {
                Error::Client(ClientError::Api {
                    status: 0,
                    message: "No choices in response".to_string(),
                })
            })?;

        let message = choice.get("message").ok_or_else(|| {
            Error::Client(ClientError::Api {
                status: 0,
                message: "No message in choice".to_string(),
            })
        })?;

        // Check for tool calls first (structured output via function calling)
        if let Some(tool_calls) = message.get("tool_calls") {
            if let Some(first_call) = tool_calls.get(0) {
                if let Some(args) = first_call.get("function").and_then(|f| f.get("arguments")) {
                    return Ok(args.as_str().unwrap_or("{}").to_string());
                }
            }
        }

        // Fall back to message content
        Ok(message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string())
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        let body = self.build_request_body(&request);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
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

        let response_json: serde_json::Value = response.json().await.map_err(ClientError::Http)?;

        let content = Self::extract_content(&response_json)?;

        let usage = response_json.get("usage").map(|u| Usage {
            prompt_tokens: u
                .get("prompt_tokens")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
            completion_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
            total_tokens: u
                .get("total_tokens")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
        });

        let finish_reason = response_json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("finish_reason"))
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
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
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

        let stream = SseStream::new(byte_stream);

        Ok(Box::pin(stream))
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}

/// Parses an SSE byte stream into ChatChunk items.
struct SseStream<S> {
    inner: S,
    buffer: String,
}

impl<S> SseStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }
}

impl<S> Stream for SseStream<S>
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
            // Check if we have a complete SSE event in the buffer
            if let Some(chunk) = parse_sse_event(&mut this.buffer) {
                return std::task::Poll::Ready(Some(Ok(chunk)));
            }

            // Try to read more data
            let inner = Pin::new(&mut this.inner);
            match inner.poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(Error::Client(ClientError::Http(e)))));
                }
                std::task::Poll::Ready(None) => {
                    // Check for remaining data in buffer
                    if let Some(chunk) = parse_sse_event(&mut this.buffer) {
                        return std::task::Poll::Ready(Some(Ok(chunk)));
                    }
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

fn parse_sse_event(buffer: &mut String) -> Option<ChatChunk> {
    loop {
        let newline_pos = buffer.find("\n\n")?;
        let event_str = buffer[..newline_pos].to_string();
        *buffer = buffer[newline_pos + 2..].to_string();

        for line in event_str.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                let data = data.trim();
                if data == "[DONE]" {
                    return None;
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let delta = json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("delta"));

                    let mut content = String::new();

                    if let Some(delta) = delta {
                        // Check for tool call deltas
                        if let Some(tool_calls) = delta.get("tool_calls") {
                            if let Some(args) = tool_calls
                                .get(0)
                                .and_then(|tc| tc.get("function"))
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                            {
                                content = args.to_string();
                            }
                        } else if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                            content = c.to_string();
                        }
                    }

                    let finish_reason = json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("finish_reason"))
                        .and_then(|f| f.as_str())
                        .map(String::from);

                    if !content.is_empty() || finish_reason.is_some() {
                        return Some(ChatChunk {
                            delta: content,
                            finish_reason,
                        });
                    }
                }
            }
        }
        // If we didn't return a chunk from this event, try the next one in the buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::types::{ChatMessage, FunctionDefinition, ToolDefinition};

    fn request_with_tool() -> ChatRequest {
        let mut request = ChatRequest::new(
            "m",
            vec![ChatMessage {
                role: "user".into(),
                content: serde_json::json!("hi"),
            }],
        );
        request.tools = Some(vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "answer".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        }]);
        request
    }

    #[test]
    fn extra_cannot_force_or_replace_tools() {
        let client = OpenAiClient::new("k");
        let mut request = request_with_tool();
        request
            .extra
            .insert("tool_choice".into(), serde_json::json!("required"));
        request.extra.insert("tools".into(), serde_json::json!([]));
        request
            .extra
            .insert("temperature".into(), serde_json::json!(0.2));
        let body = client.build_request_body(&request);
        assert!(body.get("tool_choice").is_none(), "{body}");
        assert_eq!(body["tools"][0]["function"]["name"], "answer");
        assert_eq!(body["temperature"], 0.2);
    }
}
