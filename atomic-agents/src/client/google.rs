use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::Stream;
use reqwest::Client;

use super::LlmClient;
use super::types::{ChatChunk, ChatRequest, ChatResponse, Usage};
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
            // Forcing a call (`ANY`) is avoided for the same reason as on the
            // Anthropic client: let the model choose and instruct it in the
            // system instruction to answer through the function.
            body["toolConfig"] = serde_json::json!({
                "functionCallingConfig": {
                    "mode": "AUTO"
                }
            });
            // Agents send exactly one tool, the output schema, so naming the
            // first function names the extraction function.
            if let Some(first) = tools.first() {
                let mut sys = system_instruction.take().unwrap_or_default();
                if !sys.is_empty() {
                    sys.push_str("\n\n");
                }
                sys.push_str(&format!(
                    "Respond by calling the `{}` function.",
                    first.function.name
                ));
                system_instruction = Some(sys);
            }
        }

        if let Some(sys) = system_instruction {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{"text": sys}]
            });
        }

        // Merge generation config from extra params
        let mut generation_config = serde_json::json!({});
        for (k, v) in &request.extra {
            match k.as_str() {
                "temperature" | "top_p" | "top_k" | "max_output_tokens" | "candidate_count"
                | "stop_sequences" => {
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
        format!("{}/models/{}:{}", self.base_url, model, method)
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

        let response_json: serde_json::Value = response.json().await.map_err(ClientError::Http)?;

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

        let finish_reason = gemini_finish_reason(&response_json);

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

        let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
        let byte_stream = response.bytes_stream();
        let stream = GeminiSseStream::new(byte_stream, has_tools);
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

/// The answer-bearing parts of one Gemini response (or stream chunk).
struct GeminiParts {
    /// The first `functionCall`'s arguments, serialized.
    call_args: Option<String>,
    /// Every `text` part, joined.
    text: String,
}

fn gemini_parts(response: &serde_json::Value) -> GeminiParts {
    let parts = response
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .map(Vec::as_slice)
        .unwrap_or_default();

    GeminiParts {
        call_args: parts
            .iter()
            .find_map(|p| p.get("functionCall").and_then(|fc| fc.get("args")))
            .map(|args| serde_json::to_string(args).unwrap_or_default()),
        text: parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect(),
    }
}

/// Extracts a response's content. The request no longer forces a function
/// call, so the model may answer in text, or with text around a call; a
/// `functionCall`'s arguments win whenever one is present.
fn extract_gemini_content(response: &serde_json::Value) -> String {
    let parts = gemini_parts(response);
    parts.call_args.unwrap_or(parts.text)
}

fn gemini_finish_reason(json: &serde_json::Value) -> Option<String> {
    json.get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finishReason"))
        .and_then(|f| f.as_str())
        .map(String::from)
}

/// Parses Gemini's SSE stream into ChatChunks.
///
/// With tools offered, text is held back rather than streamed: a
/// `functionCall` discards it and only the call's arguments are emitted, and
/// a response that finishes without one yields the held text on its final
/// chunk. Without tools, text streams as it arrives.
struct GeminiSseStream<S> {
    inner: S,
    buffer: String,
    has_tools: bool,
    held_text: String,
    saw_call: bool,
}

impl<S> GeminiSseStream<S> {
    fn new(inner: S, has_tools: bool) -> Self {
        Self {
            inner,
            buffer: String::new(),
            has_tools,
            held_text: String::new(),
            saw_call: false,
        }
    }

    fn on_response(&mut self, json: &serde_json::Value) -> Option<ChatChunk> {
        let GeminiParts { call_args, text } = gemini_parts(json);
        let finish_reason = gemini_finish_reason(json);

        let delta = if !self.has_tools {
            call_args.unwrap_or(text)
        } else if let Some(args) = call_args {
            self.saw_call = true;
            self.held_text.clear();
            args
        } else {
            if !self.saw_call {
                self.held_text.push_str(&text);
            }
            if finish_reason.is_some() {
                std::mem::take(&mut self.held_text)
            } else {
                String::new()
            }
        };

        (!delta.is_empty() || finish_reason.is_some()).then_some(ChatChunk {
            delta,
            finish_reason,
        })
    }

    /// Called when the byte stream ends: yields any text still held back
    /// (a stream cut off before its finishing chunk).
    fn finish(&mut self) -> Option<ChatChunk> {
        (!self.held_text.is_empty()).then(|| ChatChunk {
            delta: std::mem::take(&mut self.held_text),
            finish_reason: Some("STOP".to_string()),
        })
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
            while let Some(json) = next_gemini_event(&mut this.buffer) {
                if let Some(chunk) = this.on_response(&json) {
                    return std::task::Poll::Ready(Some(Ok(chunk)));
                }
            }

            let inner = Pin::new(&mut this.inner);
            match inner.poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(Error::Client(ClientError::Http(e)))));
                }
                std::task::Poll::Ready(None) => {
                    // A last event without its trailing blank line.
                    if !this.buffer.trim().is_empty() {
                        this.buffer.push_str("\n\n");
                        continue;
                    }
                    return std::task::Poll::Ready(this.finish().map(Ok));
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

/// Takes the next complete SSE event off `buffer` and returns its JSON data.
fn next_gemini_event(buffer: &mut String) -> Option<serde_json::Value> {
    loop {
        let newline_pos = buffer.find("\n\n")?;
        let event_str = buffer[..newline_pos].to_string();
        *buffer = buffer[newline_pos + 2..].to_string();

        let json = event_str
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .find_map(|data| serde_json::from_str::<serde_json::Value>(data.trim()).ok());
        if json.is_some() {
            return json;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::types::{ChatMessage, FunctionDefinition, ToolDefinition};
    use serde_json::json;

    fn request(with_system: bool, with_tool: bool) -> ChatRequest {
        let mut messages = Vec::new();
        if with_system {
            messages.push(ChatMessage {
                role: "system".into(),
                content: json!("You are helpful."),
            });
        }
        messages.push(ChatMessage {
            role: "user".into(),
            content: json!("Hi"),
        });
        let mut req = ChatRequest::new("gemini-2.5-flash", messages);
        if with_tool {
            req.tools = Some(vec![ToolDefinition {
                tool_type: "function".into(),
                function: FunctionDefinition {
                    name: "answer".into(),
                    description: None,
                    parameters: json!({"type": "object"}),
                    strict: None,
                },
            }]);
        }
        req
    }

    fn system_text(body: &serde_json::Value) -> &str {
        body["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .unwrap()
    }

    #[test]
    fn tool_request_uses_auto_mode_and_names_the_function() {
        let body = GoogleClient::new("k").build_request_body(&request(true, true));
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "AUTO");
        assert!(!body.to_string().contains("\"ANY\""));
        let system = system_text(&body);
        assert!(system.starts_with("You are helpful."));
        assert!(system.ends_with("Respond by calling the `answer` function."));
    }

    #[test]
    fn tool_request_without_system_message_still_names_the_function() {
        let body = GoogleClient::new("k").build_request_body(&request(false, true));
        assert_eq!(
            system_text(&body),
            "Respond by calling the `answer` function."
        );
    }

    #[test]
    fn request_without_tools_has_no_tool_config_and_unchanged_system() {
        let body = GoogleClient::new("k").build_request_body(&request(true, false));
        assert!(body.get("toolConfig").is_none());
        assert_eq!(system_text(&body), "You are helpful.");
    }

    fn response(parts: serde_json::Value) -> serde_json::Value {
        json!({"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]})
    }

    #[test]
    fn parses_text_only_response() {
        let r = response(json!([{"text": "{\"a\":"}, {"text": "1}"}]));
        assert_eq!(extract_gemini_content(&r), "{\"a\":1}");
    }

    #[test]
    fn parse_prefers_function_call_over_a_text_preamble() {
        let r = response(json!([
            {"text": "Calling the function."},
            {"functionCall": {"name": "answer", "args": {"a": 1}}}
        ]));
        assert_eq!(extract_gemini_content(&r), "{\"a\":1}");
    }

    fn collect_stream(events: &[serde_json::Value], has_tools: bool) -> String {
        use futures::StreamExt;
        let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        let pieces: Vec<Result<Bytes, reqwest::Error>> = body
            .as_bytes()
            .chunks(7)
            .map(|c| Ok(Bytes::copy_from_slice(c)))
            .collect();
        let stream = GeminiSseStream::new(futures::stream::iter(pieces), has_tools);
        futures::executor::block_on(stream.collect::<Vec<_>>())
            .into_iter()
            .map(|c| c.unwrap().delta)
            .collect()
    }

    fn chunk(parts: serde_json::Value, finish: Option<&str>) -> serde_json::Value {
        let mut candidate = json!({"content": {"role": "model", "parts": parts}});
        if let Some(f) = finish {
            candidate["finishReason"] = json!(f);
        }
        json!({"candidates": [candidate]})
    }

    #[test]
    fn stream_with_tools_drops_a_text_preamble_before_the_function_call() {
        let events = [
            chunk(json!([{"text": "Let me call "}]), None),
            chunk(json!([{"text": "the function."}]), None),
            chunk(
                json!([{"functionCall": {"name": "answer", "args": {"a": 1}}}]),
                Some("STOP"),
            ),
        ];
        assert_eq!(collect_stream(&events, true), "{\"a\":1}");
    }

    #[test]
    fn stream_with_tools_and_text_only_yields_the_text_at_the_end() {
        let events = [
            chunk(json!([{"text": "{\"a\":"}]), None),
            chunk(json!([{"text": "1}"}]), Some("STOP")),
        ];
        assert_eq!(collect_stream(&events, true), "{\"a\":1}");
    }

    #[test]
    fn stream_without_tools_streams_text_as_it_arrives() {
        use futures::StreamExt;
        let body: String = [
            chunk(json!([{"text": "Hel"}]), None),
            chunk(json!([{"text": "lo"}]), Some("STOP")),
        ]
        .iter()
        .map(|e| format!("data: {e}\n\n"))
        .collect();
        let stream = GeminiSseStream::new(
            futures::stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(body))]),
            false,
        );
        let deltas: Vec<String> = futures::executor::block_on(stream.collect::<Vec<_>>())
            .into_iter()
            .map(|c| c.unwrap().delta)
            .collect();
        assert_eq!(deltas, ["Hel", "lo"]);
    }
}
