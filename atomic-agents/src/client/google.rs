use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;

use super::types::{ChatChunk, ChatRequest, ChatResponse, Usage};
use super::LlmClient;
use crate::error::{ClientError, Error};

/// Client for Google Gemini (Generative Language API).
///
/// Handles Gemini-specific conventions: "model" role instead of "assistant",
/// "user" role for mid-conversation context injection, and different tool format.
pub struct GoogleClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl GoogleClient {
    /// Creates a new Google Gemini client.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
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
        let mut system_instruction = None;
        let mut contents = Vec::new();

        for msg in &request.messages {
            let role = match msg.role.as_str() {
                "system" => {
                    // Gemini uses systemInstruction as top-level param
                    if let Some(text) = msg.content.as_str() {
                        system_instruction = Some(text.to_string());
                    }
                    continue;
                }
                "assistant" => "model",
                other => other,
            };

            let parts = match &msg.content {
                serde_json::Value::String(s) => {
                    vec![serde_json::json!({"text": s})]
                }
                serde_json::Value::Array(arr) => arr.clone(),
                other => vec![serde_json::json!({"text": other.to_string()})],
            };

            contents.push(serde_json::json!({
                "role": role,
                "parts": parts,
            }));
        }

        let mut body = serde_json::json!({
            "contents": contents,
        });

        if let Some(sys) = system_instruction {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{"text": sys}]
            });
        }

        // Convert tools to Gemini format
        if let Some(tools) = &request.tools {
            let function_declarations: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    let mut decl = serde_json::json!({
                        "name": t.function.name,
                        "parameters": t.function.parameters,
                    });
                    if let Some(desc) = &t.function.description {
                        decl["description"] = serde_json::json!(desc);
                    }
                    decl
                })
                .collect();
            body["tools"] = serde_json::json!([{
                "functionDeclarations": function_declarations,
            }]);
            body["toolConfig"] = serde_json::json!({
                "functionCallingConfig": {
                    "mode": "ANY"
                }
            });
        }

        // Merge generation config from extra params
        let mut generation_config = serde_json::json!({});
        for (k, v) in &request.extra {
            match k.as_str() {
                "temperature" | "top_p" | "top_k" | "max_output_tokens"
                | "candidate_count" | "stop_sequences" => {
                    generation_config[k] = v.clone();
                }
                _ => {
                    body[k] = v.clone();
                }
            }
        }
        if generation_config != serde_json::json!({}) {
            body["generationConfig"] = generation_config;
        }

        body
    }

    fn endpoint_url(&self, model: &str, stream: bool) -> String {
        let method = if stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        format!(
            "{}/models/{}:{}",
            self.base_url, model, method
        )
    }
}

#[async_trait]
impl LlmClient for GoogleClient {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        let body = self.build_request_body(&request);
        let url = format!(
            "{}?key={}",
            self.endpoint_url(&request.model, false),
            self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ClientError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            if status.as_u16() == 401 || status.as_u16() == 403 {
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

        // Extract content from Gemini response
        let content = extract_gemini_content(&response_json);

        let usage = response_json.get("usageMetadata").map(|u| Usage {
            prompt_tokens: u
                .get("promptTokenCount")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
            completion_tokens: u
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
            total_tokens: u
                .get("totalTokenCount")
                .and_then(|v| v.as_u64().map(|n| n as u32)),
        });

        let finish_reason = response_json
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("finishReason"))
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
        let body = self.build_request_body(&request);
        let url = format!(
            "{}?key={}",
            self.endpoint_url(&request.model, true),
            self.api_key
        );

        let response = self
            .client
            .post(&url)
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
        let stream = GeminiSseStream::new(byte_stream);
        Ok(Box::pin(stream))
    }

    fn provider_name(&self) -> &str {
        "google"
    }

    fn assistant_role(&self) -> &str {
        "model"
    }

    fn tool_result_role(&self) -> &str {
        "user"
    }
}

fn extract_gemini_content(response: &serde_json::Value) -> String {
    let candidate = response
        .get("candidates")
        .and_then(|c| c.get(0));

    let parts = candidate
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array());

    if let Some(parts) = parts {
        for part in parts {
            // Check for function call (tool use)
            if let Some(fc) = part.get("functionCall") {
                if let Some(args) = fc.get("args") {
                    return serde_json::to_string(args).unwrap_or_default();
                }
            }
            // Text content
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                return text.to_string();
            }
        }
    }
    String::new()
}

struct GeminiSseStream<S> {
    inner: S,
    buffer: String,
}

impl<S> GeminiSseStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }
}

impl<S> Stream for GeminiSseStream<S>
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
            if let Some(chunk) = parse_gemini_sse(&mut this.buffer) {
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
                    if let Some(chunk) = parse_gemini_sse(&mut this.buffer) {
                        return std::task::Poll::Ready(Some(Ok(chunk)));
                    }
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

fn parse_gemini_sse(buffer: &mut String) -> Option<ChatChunk> {
    loop {
        let newline_pos = buffer.find("\n\n")?;
        let event_str = buffer[..newline_pos].to_string();
        *buffer = buffer[newline_pos + 2..].to_string();

        for line in event_str.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                let data = data.trim();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let content = extract_gemini_content(&json);
                    let finish_reason = json
                        .get("candidates")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("finishReason"))
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
    }
}
