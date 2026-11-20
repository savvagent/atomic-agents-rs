use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::Stream;
use reqwest::Client;

use super::LlmClient;
use super::types::{ChatChunk, ChatRequest, ChatResponse, Usage};
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
            // Current Claude models reject a forced tool_choice (`any`/`tool`)
            // with a 400, so let the model choose and instruct it in the
            // system prompt to answer through the tool.
            body["tool_choice"] = serde_json::json!({"type": "auto"});
            // Agents send exactly one tool, the output schema, so naming the
            // first tool names the extraction tool.
            if let Some(first) = tools.first() {
                if !system_text.is_empty() {
                    system_text.push_str("\n\n");
                }
                system_text.push_str(&format!(
                    "Respond by calling the `{}` tool.",
                    first.function.name
                ));
                body["system"] = serde_json::json!(system_text);
            }
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

        let response_json: serde_json::Value = response.json().await.map_err(ClientError::Http)?;

        parse_response(&response_json)
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

        let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
        let byte_stream = response.bytes_stream();
        let stream = AnthropicSseStream::new(byte_stream, has_tools);
        Ok(Box::pin(stream))
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

/// Converts an Anthropic Messages API response body into a [`ChatResponse`].
///
/// The request no longer forces a tool call, so the model may answer with a
/// plain `text` block instead of a `tool_use` block, or with both (a text
/// preamble or epilogue around the call). A `tool_use` block's input wins
/// whenever one is present; otherwise the text blocks are joined.
fn parse_response(json: &serde_json::Value) -> Result<ChatResponse, Error> {
    let blocks = json
        .get("content")
        .and_then(|c| c.as_array())
        .map(Vec::as_slice)
        .unwrap_or_default();
    let is =
        |b: &serde_json::Value, kind: &str| b.get("type").and_then(|t| t.as_str()) == Some(kind);

    let tool_input = blocks
        .iter()
        .filter(|b| is(b, "tool_use"))
        .find_map(|b| b.get("input"));

    let content = match tool_input {
        Some(input) => serde_json::to_string(input).unwrap_or_default(),
        None => blocks
            .iter()
            .filter(|b| is(b, "text"))
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect(),
    };

    let usage = json.get("usage").map(|u| Usage {
        prompt_tokens: u
            .get("input_tokens")
            .and_then(|v| v.as_u64().map(|n| n as u32)),
        completion_tokens: u
            .get("output_tokens")
            .and_then(|v| v.as_u64().map(|n| n as u32)),
        total_tokens: None, // Anthropic doesn't return total
    });

    let finish_reason = json
        .get("stop_reason")
        .and_then(|f| f.as_str())
        .map(String::from);

    Ok(ChatResponse {
        content,
        finish_reason,
        usage,
    })
}

/// Parses Anthropic's SSE stream format into ChatChunks.
///
/// The request no longer forces a tool call, so when tools are offered the
/// model may write a text preamble before its `tool_use` block. Concatenating
/// that text with the tool's `input_json_delta` fragments would produce
/// invalid JSON, so with tools the stream holds text back: a `tool_use` block
/// discards it and only the tool input is streamed, and a message that ends
/// without one yields the held text once, on its final chunk. Without tools,
/// text deltas stream as they arrive.
struct AnthropicSseStream<S> {
    inner: S,
    buffer: String,
    state: ToolStreamState,
}

impl<S> AnthropicSseStream<S> {
    fn new(inner: S, has_tools: bool) -> Self {
        Self {
            inner,
            buffer: String::new(),
            state: ToolStreamState::new(has_tools),
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
            while let Some(event) = next_anthropic_event(&mut this.buffer) {
                if let Some(chunk) = this.state.on_event(event) {
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
                    return std::task::Poll::Ready(this.state.finish().map(Ok));
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

/// The parts of an Anthropic stream event the client acts on.
#[derive(Debug, PartialEq)]
enum AnthropicEvent {
    Text(String),
    ToolJson(String),
    ToolUseStart,
    Stop(String),
}

/// Applies the hold-text-while-tools-are-offered rule to stream events.
#[derive(Debug)]
struct ToolStreamState {
    has_tools: bool,
    held_text: String,
    saw_tool_use: bool,
    done: bool,
}

impl ToolStreamState {
    fn new(has_tools: bool) -> Self {
        Self {
            has_tools,
            held_text: String::new(),
            saw_tool_use: false,
            done: false,
        }
    }

    fn on_event(&mut self, event: AnthropicEvent) -> Option<ChatChunk> {
        let chunk = |delta: String, finish_reason: Option<String>| {
            Some(ChatChunk {
                delta,
                finish_reason,
            })
        };
        match event {
            AnthropicEvent::Text(text) if !self.has_tools => chunk(text, None),
            AnthropicEvent::Text(text) => {
                if !self.saw_tool_use {
                    self.held_text.push_str(&text);
                }
                None
            }
            AnthropicEvent::ToolUseStart => {
                self.saw_tool_use = true;
                self.held_text.clear();
                None
            }
            AnthropicEvent::ToolJson(json) => chunk(json, None),
            AnthropicEvent::Stop(reason) => {
                self.done = true;
                chunk(std::mem::take(&mut self.held_text), Some(reason))
            }
        }
    }

    /// Called when the byte stream ends: yields any text still held back
    /// (a stream cut off before its stop events).
    fn finish(&mut self) -> Option<ChatChunk> {
        if self.held_text.is_empty() {
            return None;
        }
        Some(ChatChunk {
            delta: std::mem::take(&mut self.held_text),
            finish_reason: if self.done {
                None
            } else {
                Some("end_turn".to_string())
            },
        })
    }
}

/// Takes the next complete event off `buffer` that the client acts on.
fn next_anthropic_event(buffer: &mut String) -> Option<AnthropicEvent> {
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

        let json = serde_json::from_str::<serde_json::Value>(data_str).ok();

        match event_type {
            "content_block_start" => {
                let block_type = json
                    .as_ref()
                    .and_then(|j| j.get("content_block"))
                    .and_then(|b| b.get("type"))
                    .and_then(|t| t.as_str());
                if block_type == Some("tool_use") {
                    return Some(AnthropicEvent::ToolUseStart);
                }
            }
            "content_block_delta" => {
                let Some(delta) = json.as_ref().and_then(|j| j.get("delta")) else {
                    continue;
                };
                let field = |name: &str| {
                    delta
                        .get(name)
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string()
                };
                let event = match delta.get("type").and_then(|t| t.as_str()) {
                    Some("text_delta") => AnthropicEvent::Text(field("text")),
                    Some("input_json_delta") => AnthropicEvent::ToolJson(field("partial_json")),
                    _ => continue,
                };
                if matches!(&event, AnthropicEvent::Text(t) | AnthropicEvent::ToolJson(t) if t.is_empty())
                {
                    continue;
                }
                return Some(event);
            }
            "message_delta" => {
                let stop_reason = json
                    .as_ref()
                    .and_then(|j| j.get("delta"))
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|s| s.as_str());
                if let Some(reason) = stop_reason {
                    return Some(AnthropicEvent::Stop(reason.to_string()));
                }
            }
            "message_stop" => return Some(AnthropicEvent::Stop("end_turn".to_string())),
            _ => {
                // Skip other event types (message_start, ping, content_block_stop, etc.)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::types::{ChatMessage, FunctionDefinition, ToolDefinition};
    use serde_json::json;

    fn request(with_tool: bool) -> ChatRequest {
        let mut req = ChatRequest::new(
            "claude-sonnet-4-5",
            vec![
                ChatMessage {
                    role: "system".into(),
                    content: json!("You are helpful."),
                },
                ChatMessage {
                    role: "user".into(),
                    content: json!("Hi"),
                },
            ],
        );
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

    #[test]
    fn tool_request_uses_auto_tool_choice_and_names_the_tool() {
        let body = AnthropicClient::new("k").build_request_body(&request(true));
        assert_eq!(body["tool_choice"], json!({"type": "auto"}));
        let system = body["system"].as_str().unwrap();
        assert!(system.starts_with("You are helpful."));
        assert!(system.ends_with("Respond by calling the `answer` tool."));
    }

    #[test]
    fn request_without_tools_has_no_tool_choice_and_unchanged_system() {
        let body = AnthropicClient::new("k").build_request_body(&request(false));
        assert!(body.get("tool_choice").is_none());
        assert_eq!(body["system"], json!("You are helpful."));
    }

    #[test]
    fn parses_text_only_response() {
        let r =
            parse_response(&json!({"content": [{"type": "text", "text": "{\"a\":1}"}]})).unwrap();
        assert_eq!(r.content, "{\"a\":1}");
    }

    #[test]
    fn parses_tool_use_response() {
        let r = parse_response(&json!({"content": [
            {"type": "tool_use", "id": "t", "name": "answer", "input": {"a": 1}}
        ]}))
        .unwrap();
        assert_eq!(r.content, "{\"a\":1}");
    }

    #[test]
    fn parse_prefers_tool_use_over_a_later_text_block() {
        let r = parse_response(&json!({"content": [
            {"type": "tool_use", "id": "t", "name": "answer", "input": {"a": 1}},
            {"type": "text", "text": "Here is the answer."}
        ]}))
        .unwrap();
        assert_eq!(r.content, "{\"a\":1}");
    }

    fn sse(events: &[(&str, serde_json::Value)]) -> String {
        events
            .iter()
            .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
            .collect()
    }

    fn text_delta(text: &str) -> (&'static str, serde_json::Value) {
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": text}}),
        )
    }

    fn json_delta(partial: &str) -> (&'static str, serde_json::Value) {
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "input_json_delta", "partial_json": partial}}),
        )
    }

    fn end(stop_reason: &str) -> Vec<(&'static str, serde_json::Value)> {
        vec![
            (
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": stop_reason}}),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ]
    }

    /// Feeds `body` through the SSE parser in small pieces (so events split
    /// across reads) and returns the concatenated deltas.
    fn collect_stream(body: &str, has_tools: bool) -> String {
        use futures::StreamExt;
        let pieces: Vec<Result<Bytes, reqwest::Error>> = body
            .as_bytes()
            .chunks(7)
            .map(|c| Ok(Bytes::copy_from_slice(c)))
            .collect();
        let stream = AnthropicSseStream::new(futures::stream::iter(pieces), has_tools);
        futures::executor::block_on(stream.collect::<Vec<_>>())
            .into_iter()
            .map(|c| c.unwrap().delta)
            .collect()
    }

    #[test]
    fn stream_with_tools_drops_a_text_preamble_before_tool_use() {
        let mut events = vec![
            (
                "message_start",
                json!({"type": "message_start", "message": {"id": "m"}}),
            ),
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}}),
            ),
            text_delta("Let me answer "),
            text_delta("with the tool."),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 1,
                       "content_block": {"type": "tool_use", "id": "t", "name": "answer", "input": {}}}),
            ),
            json_delta("{\"a\""),
            json_delta(":1}"),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 1}),
            ),
        ];
        events.extend(end("tool_use"));
        let out = collect_stream(&sse(&events), true);
        assert_eq!(out, "{\"a\":1}");
        serde_json::from_str::<serde_json::Value>(&out).unwrap();
    }

    #[test]
    fn stream_with_tools_and_text_only_yields_the_text_once_at_the_end() {
        let mut events = vec![
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}}),
            ),
            text_delta("{\"a\""),
            text_delta(":1}"),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
        ];
        events.extend(end("end_turn"));
        assert_eq!(collect_stream(&sse(&events), true), "{\"a\":1}");
    }

    #[test]
    fn stream_with_tools_flushes_held_text_when_the_stream_ends_early() {
        let events = vec![text_delta("partial")];
        assert_eq!(collect_stream(&sse(&events), true), "partial");
    }

    #[test]
    fn stream_without_tools_streams_text_deltas_as_they_arrive() {
        use futures::StreamExt;
        let mut events = vec![text_delta("Hel"), text_delta("lo")];
        events.extend(end("end_turn"));
        let body = sse(&events);
        let stream = AnthropicSseStream::new(
            futures::stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(body))]),
            false,
        );
        let chunks: Vec<ChatChunk> = futures::executor::block_on(stream.collect::<Vec<_>>())
            .into_iter()
            .map(Result::unwrap)
            .collect();
        assert_eq!(chunks[0].delta, "Hel");
        assert_eq!(chunks[1].delta, "lo");
        assert_eq!(
            chunks.iter().map(|c| c.delta.as_str()).collect::<String>(),
            "Hello"
        );
    }
}
