//! The LLM side of the guest: [`LlmHost`], its data types and [`HostLlmClient`].

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::client::LlmClient;
use crate::client::types::{ChatChunk, ChatRequest, ChatResponse, Usage as ChatUsage};
use crate::error::{ClientError, Error};

/// Who wrote a [`Message`]. System text travels in
/// [`CompletionRequest::system`], not as a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// Text from the user.
    User,
    /// Text or tool calls from the model.
    Assistant,
}

/// A model's request to run a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Identifier that pairs this call with its [`ToolResult`].
    pub id: String,
    /// Name of the tool to run.
    pub name: String,
    /// Arguments, as a JSON document.
    pub arguments: String,
}

/// The outcome of a [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The [`ToolCall::id`] this answers.
    pub call_id: String,
    /// The tool's output.
    pub content: String,
    /// Whether the tool failed and `content` describes the failure.
    pub is_error: bool,
}

/// One piece of a [`Message`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ContentBlock {
    /// Plain text.
    Text(String),
    /// A tool call made by the model.
    ToolCall(ToolCall),
    /// The result of a tool call.
    ToolResult(ToolResult),
}

/// A conversation turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who wrote it.
    pub role: Role,
    /// What it says.
    pub content: Vec<ContentBlock>,
}

/// A tool the model may call, as sent to the host.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// What the tool does.
    pub description: String,
    /// JSON Schema of the arguments, as a JSON document.
    pub input_schema: String,
}

/// A request for one model completion.
///
/// There is deliberately no field that selects or forces a tool: the host
/// decides how structured output is obtained from [`output_schema`](Self::output_schema).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier.
    pub model: String,
    /// System prompt, if any.
    pub system: Option<String>,
    /// The conversation so far.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    pub tools: Vec<ToolSpec>,
    /// JSON Schema the reply must follow, as a JSON document.
    pub output_schema: Option<String>,
    /// Upper bound on generated tokens.
    pub max_tokens: Option<u32>,
    /// Further sampling parameters, as a JSON object document. Never holds
    /// a [`TOOL_FORCING_KEYS`] key from [`HostLlmClient`]; a host must not
    /// honour one from any guest.
    pub params: Option<String>,
}

/// Parameter keys that select or force a tool, or pick the output format.
///
/// The host owns these: it decides how structured output is obtained from
/// [`CompletionRequest::output_schema`], and a forced tool call lets
/// whoever writes the parameters steer the model past its instructions. So
/// [`HostLlmClient`] rejects them in a request's extra parameters, and a
/// guest that forwards parameters from untrusted input (such as a pipeline
/// spec) should reject them too. Matching is case-sensitive: these are the
/// spellings the providers read.
pub const TOOL_FORCING_KEYS: [&str; 9] = [
    "tool_choice",
    "tool_config",
    "toolConfig",
    "function_call",
    "functionCallingConfig",
    "tools",
    "output_config",
    "output_format",
    "response_format",
];

/// Token counts for one completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub input_tokens: u32,
    /// Tokens generated.
    pub output_tokens: u32,
}

/// The host's reply to a [`CompletionRequest`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Completion {
    /// Generated text, if any. With an `output_schema` this is the JSON reply.
    pub text: Option<String>,
    /// Tool calls the model made.
    pub tool_calls: Vec<ToolCall>,
    /// Why generation stopped, as the host reports it.
    pub stop_reason: String,
    /// Token counts, when the host reports them.
    pub usage: Option<Usage>,
}

/// Why the host could not complete a request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmError {
    /// The host declined to serve the request.
    Refused(String),
    /// A quota or limit was reached.
    LimitExceeded(String),
    /// The request was malformed.
    InvalidRequest(String),
    /// The model is temporarily unreachable.
    Unavailable(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(m) => write!(f, "refused: {m}"),
            Self::LimitExceeded(m) => write!(f, "limit exceeded: {m}"),
            Self::InvalidRequest(m) => write!(f, "invalid request: {m}"),
            Self::Unavailable(m) => write!(f, "unavailable: {m}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Maps to [`Error::Client`] as [`ClientError::Api`] with `status` 0: the
/// failure did not come over HTTP, so there is no real status code.
impl From<LlmError> for Error {
    fn from(e: LlmError) -> Self {
        Error::Client(ClientError::Api {
            status: 0,
            message: format!("host llm: {e}"),
        })
    }
}

/// The model the host provides to the guest.
///
/// The call is synchronous. Implemented for `&T` and `Arc<T>` whenever `T`
/// implements it.
pub trait LlmHost: Send + Sync {
    /// Runs one completion.
    fn complete(&self, req: CompletionRequest) -> Result<Completion, LlmError>;
}

impl<T: LlmHost + ?Sized> LlmHost for &T {
    fn complete(&self, req: CompletionRequest) -> Result<Completion, LlmError> {
        (**self).complete(req)
    }
}

impl<T: LlmHost + ?Sized> LlmHost for Arc<T> {
    fn complete(&self, req: CompletionRequest) -> Result<Completion, LlmError> {
        (**self).complete(req)
    }
}

/// An [`LlmClient`] backed by an [`LlmHost`].
///
/// The framework's single extraction tool is sent as
/// [`CompletionRequest::output_schema`] with no tools; `system` messages are
/// joined into [`CompletionRequest::system`]; `max_tokens` is lifted out of
/// the request's extra parameters and the rest go to
/// [`CompletionRequest::params`]. An extra parameter named in
/// [`TOOL_FORCING_KEYS`] is an [`LlmError::InvalidRequest`], because the host,
/// not the caller, decides how a tool is chosen.
///
/// ```
/// use atomic_agents::client::LlmClient;
/// use atomic_agents::guest::{Completion, CompletionRequest, HostLlmClient, LlmError, LlmHost};
///
/// struct Echo;
/// impl LlmHost for Echo {
///     fn complete(&self, _req: CompletionRequest) -> Result<Completion, LlmError> {
///         Ok(Completion {
///             text: Some("hi".into()),
///             tool_calls: vec![],
///             stop_reason: "end_turn".into(),
///             usage: None,
///         })
///     }
/// }
///
/// assert_eq!(HostLlmClient::new(Echo).provider_name(), "host");
/// ```
#[derive(Clone, Debug)]
pub struct HostLlmClient<H: LlmHost> {
    host: H,
}

impl<H: LlmHost> HostLlmClient<H> {
    /// Wraps `host`.
    pub fn new(host: H) -> Self {
        Self { host }
    }

    fn to_completion_request(request: ChatRequest) -> Result<CompletionRequest, Error> {
        let mut system: Vec<String> = Vec::new();
        let mut messages = Vec::new();
        for m in request.messages {
            let text = match m.content {
                Value::String(s) => s,
                other => other.to_string(),
            };
            match m.role.as_str() {
                "system" => system.push(text),
                // Roles other than system/assistant (tool, function, ...) carry
                // text the model must read, so they go in as user turns.
                role => messages.push(Message {
                    role: if role == "assistant" {
                        Role::Assistant
                    } else {
                        Role::User
                    },
                    content: vec![ContentBlock::Text(text)],
                }),
            }
        }

        // The framework sends exactly one tool: the extraction tool. Its
        // parameters are the schema the reply must follow. Anything else
        // cannot be expressed to the host, so it is an error, not a silent drop.
        let output_schema = match request.tools.as_deref() {
            None | Some([]) => None,
            Some([extraction]) => Some(serde_json::to_string(&extraction.function.parameters)?),
            Some(many) => {
                return Err(LlmError::InvalidRequest(format!(
                    "{} tools given; only the single extraction tool is supported",
                    many.len()
                ))
                .into());
            }
        };

        let mut extra: Map<String, Value> = request.extra.into_iter().collect();
        // The host owns tool choice and output format; see TOOL_FORCING_KEYS.
        if let Some(key) = TOOL_FORCING_KEYS.iter().find(|k| extra.contains_key(**k)) {
            return Err(LlmError::InvalidRequest(format!(
                "parameter `{key}` is reserved for the host"
            ))
            .into());
        }
        let max_tokens = match extra.remove("max_tokens") {
            None => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        LlmError::InvalidRequest(format!(
                            "max_tokens must be an integer from 0 to {}, got {value}",
                            u32::MAX
                        ))
                    })?,
            ),
        };
        let params = if extra.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&extra)?)
        };

        Ok(CompletionRequest {
            model: request.model,
            system: if system.is_empty() {
                None
            } else {
                Some(system.join("\n"))
            },
            messages,
            tools: Vec::new(),
            output_schema,
            max_tokens,
            params,
        })
    }

    fn complete(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        let completion = self.host.complete(Self::to_completion_request(request)?)?;
        Ok(ChatResponse {
            content: completion.text.unwrap_or_default(),
            finish_reason: Some(completion.stop_reason),
            usage: completion.usage.map(|u| ChatUsage {
                prompt_tokens: Some(u.input_tokens),
                completion_tokens: Some(u.output_tokens),
                total_tokens: Some(u.input_tokens.saturating_add(u.output_tokens)),
            }),
        })
    }
}

#[async_trait]
impl<H: LlmHost> LlmClient for HostLlmClient<H> {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        self.complete(request)
    }

    /// The host call is not incremental, so the stream is the whole reply as
    /// one chunk.
    async fn chat_completion_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, Error>> + Send>>, Error> {
        let response = self.complete(request)?;
        let chunk = ChatChunk {
            delta: response.content,
            finish_reason: response.finish_reason,
        };
        Ok(Box::pin(futures::stream::iter([Ok(chunk)])))
    }

    fn provider_name(&self) -> &str {
        "host"
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};

    use futures::StreamExt;
    use futures::executor::block_on;
    use serde_json::json;

    use super::*;
    use crate::agent::{AgentConfig, AtomicAgent};
    use crate::client::LlmClient;
    use crate::client::types::{
        ChatMessage, ChatRequest, FunctionDefinition, ToolDefinition, Usage as ChatUsage,
    };
    use crate::error::{ClientError, Error};
    use crate::schema::{BasicChatInputSchema, BasicChatOutputSchema};

    /// Records every request and replays scripted completions.
    #[derive(Default)]
    struct MockLlm {
        requests: Mutex<Vec<CompletionRequest>>,
        script: Mutex<VecDeque<Result<Completion, LlmError>>>,
    }

    impl MockLlm {
        fn scripted(items: Vec<Result<Completion, LlmError>>) -> Arc<Self> {
            Arc::new(Self {
                requests: Mutex::default(),
                script: Mutex::new(items.into()),
            })
        }

        fn requests(&self) -> Vec<CompletionRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl LlmHost for MockLlm {
        fn complete(&self, req: CompletionRequest) -> Result<Completion, LlmError> {
            self.requests.lock().unwrap().push(req);
            self.script
                .lock()
                .unwrap()
                .pop_front()
                .expect("script exhausted")
        }
    }

    fn text(t: &str) -> Completion {
        Completion {
            text: Some(t.to_string()),
            tool_calls: vec![],
            stop_reason: "end_turn".to_string(),
            usage: Some(Usage {
                input_tokens: 3,
                output_tokens: 4,
            }),
        }
    }

    fn msg(role: &str, content: serde_json::Value) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content,
        }
    }

    fn extraction_tool(parameters: serde_json::Value) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "Out".to_string(),
                description: None,
                parameters,
                strict: None,
            },
        }
    }

    #[test]
    fn system_messages_join_in_any_position() {
        let host = MockLlm::scripted(vec![Ok(text("ok"))]);
        let client = HostLlmClient::new(host.clone());
        let request = ChatRequest::new(
            "m",
            vec![
                msg("system", json!("one")),
                msg("user", json!("hi")),
                msg("system", json!("two")),
                msg("assistant", json!("yo")),
            ],
        );
        block_on(client.chat_completion(request)).unwrap();
        let sent = &host.requests()[0];
        assert_eq!(sent.model, "m");
        assert_eq!(sent.system.as_deref(), Some("one\ntwo"));
        assert_eq!(sent.messages.len(), 2);
        assert_eq!(sent.messages[0].role, Role::User);
        assert_eq!(sent.messages[1].role, Role::Assistant);
        assert_eq!(
            sent.messages[1].content,
            vec![ContentBlock::Text("yo".to_string())]
        );
    }

    #[test]
    fn no_system_message_means_no_system() {
        let host = MockLlm::scripted(vec![Ok(text("ok"))]);
        let client = HostLlmClient::new(host.clone());
        block_on(client.chat_completion(ChatRequest::new("m", vec![msg("user", json!("hi"))])))
            .unwrap();
        assert_eq!(host.requests()[0].system, None);
    }

    #[test]
    fn unknown_roles_become_user_and_non_string_content_is_serialized() {
        let host = MockLlm::scripted(vec![Ok(text("ok"))]);
        let client = HostLlmClient::new(host.clone());
        let parts = json!([{"type": "text", "text": "a"}]);
        let request = ChatRequest::new("m", vec![msg("tool", parts.clone())]);
        block_on(client.chat_completion(request)).unwrap();
        let sent = &host.requests()[0];
        assert_eq!(sent.messages[0].role, Role::User);
        assert_eq!(
            sent.messages[0].content,
            vec![ContentBlock::Text(parts.to_string())]
        );
    }

    #[test]
    fn extraction_tool_becomes_output_schema_and_no_tools() {
        let host = MockLlm::scripted(vec![Ok(text("ok"))]);
        let client = HostLlmClient::new(host.clone());
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let mut request = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
        request.tools = Some(vec![extraction_tool(schema.clone())]);
        block_on(client.chat_completion(request)).unwrap();
        let sent = &host.requests()[0];
        assert!(sent.tools.is_empty());
        let sent_schema: serde_json::Value =
            serde_json::from_str(sent.output_schema.as_deref().unwrap()).unwrap();
        assert_eq!(sent_schema, schema);
    }

    #[test]
    fn max_tokens_and_params_are_split() {
        let host = MockLlm::scripted(vec![Ok(text("ok")), Ok(text("ok"))]);
        let client = HostLlmClient::new(host.clone());

        let mut request = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
        request.extra = HashMap::from([
            ("max_tokens".to_string(), json!(256)),
            ("temperature".to_string(), json!(0.5)),
        ]);
        block_on(client.chat_completion(request)).unwrap();

        let mut only_max = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
        only_max.extra = HashMap::from([("max_tokens".to_string(), json!(10))]);
        block_on(client.chat_completion(only_max)).unwrap();

        let sent = host.requests();
        assert_eq!(sent[0].max_tokens, Some(256));
        let params: serde_json::Value =
            serde_json::from_str(sent[0].params.as_deref().unwrap()).unwrap();
        assert_eq!(params, json!({"temperature": 0.5}));
        assert_eq!(sent[1].max_tokens, Some(10));
        assert_eq!(sent[1].params, None);
    }

    fn invalid_request(result: Result<ChatResponse, Error>) -> String {
        match result {
            Err(Error::Client(ClientError::Api { status: 0, message })) => message,
            other => panic!("expected host llm InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn more_than_one_tool_is_rejected_not_dropped() {
        let host = MockLlm::scripted(vec![]);
        let client = HostLlmClient::new(host.clone());
        let mut request = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
        request.tools = Some(vec![extraction_tool(json!({})), extraction_tool(json!({}))]);
        let message = invalid_request(block_on(client.chat_completion(request)));
        assert!(
            message.starts_with("host llm: invalid request"),
            "{message}"
        );
        assert!(host.requests().is_empty(), "nothing reaches the host");
    }

    #[test]
    fn tool_forcing_keys_in_extra_are_rejected() {
        for key in [
            "tool_choice",
            "tool_config",
            "toolConfig",
            "function_call",
            "functionCallingConfig",
            "tools",
            "output_config",
            "output_format",
            "response_format",
        ] {
            let host = MockLlm::scripted(vec![]);
            let client = HostLlmClient::new(host.clone());
            let mut request = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
            request.extra = HashMap::from([(key.to_string(), json!({"type": "any"}))]);
            let message = invalid_request(block_on(client.chat_completion(request)));
            assert!(message.contains(key), "{key}: {message}");
            assert!(host.requests().is_empty(), "{key} reached the host");
        }
    }

    #[test]
    fn empty_tools_means_no_output_schema() {
        let host = MockLlm::scripted(vec![Ok(text("ok"))]);
        let client = HostLlmClient::new(host.clone());
        let mut request = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
        request.tools = Some(vec![]);
        block_on(client.chat_completion(request)).unwrap();
        assert_eq!(host.requests()[0].output_schema, None);
    }

    #[test]
    fn max_tokens_that_is_not_a_u32_is_rejected() {
        for bad in [json!(5_000_000_000u64), json!(-1), json!("10"), json!(1.5)] {
            let host = MockLlm::scripted(vec![]);
            let client = HostLlmClient::new(host.clone());
            let mut request = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
            request.extra = HashMap::from([("max_tokens".to_string(), bad.clone())]);
            let message = invalid_request(block_on(client.chat_completion(request)));
            assert!(message.contains("max_tokens"), "{message}");
            assert!(message.contains(&bad.to_string()), "{message}");
            assert!(host.requests().is_empty());
        }
    }

    #[test]
    fn response_maps_text_stop_reason_and_usage() {
        let host = MockLlm::scripted(vec![
            Ok(text("hello")),
            Ok(Completion {
                text: None,
                tool_calls: vec![],
                stop_reason: "max_tokens".to_string(),
                usage: None,
            }),
        ]);
        let client = HostLlmClient::new(host);
        let req = || ChatRequest::new("m", vec![msg("user", json!("hi"))]);
        let first = block_on(client.chat_completion(req())).unwrap();
        assert_eq!(first.content, "hello");
        assert_eq!(first.finish_reason.as_deref(), Some("end_turn"));
        let usage: ChatUsage = first.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(3));
        assert_eq!(usage.completion_tokens, Some(4));
        assert_eq!(usage.total_tokens, Some(7));
        let second = block_on(client.chat_completion(req())).unwrap();
        assert_eq!(second.content, "");
        assert_eq!(second.finish_reason.as_deref(), Some("max_tokens"));
        assert!(second.usage.is_none());
    }

    #[test]
    fn stream_yields_one_chunk_then_ends() {
        let host = MockLlm::scripted(vec![Ok(text("whole"))]);
        let client = HostLlmClient::new(host);
        let request = ChatRequest::new("m", vec![msg("user", json!("hi"))]);
        let chunks: Vec<_> = block_on(async {
            client
                .chat_completion_stream(request)
                .await
                .unwrap()
                .collect()
                .await
        });
        assert_eq!(chunks.len(), 1);
        let chunk = chunks[0].as_ref().unwrap();
        assert_eq!(chunk.delta, "whole");
        assert_eq!(chunk.finish_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn provider_name_is_host() {
        let client = HostLlmClient::new(MockLlm::scripted(vec![]));
        assert_eq!(client.provider_name(), "host");
    }

    #[test]
    fn host_errors_map_to_client_api_errors_with_status_zero() {
        let cases = [
            (LlmError::Refused("r".into()), "host llm: refused: r"),
            (
                LlmError::LimitExceeded("l".into()),
                "host llm: limit exceeded: l",
            ),
            (
                LlmError::InvalidRequest("i".into()),
                "host llm: invalid request: i",
            ),
            (
                LlmError::Unavailable("u".into()),
                "host llm: unavailable: u",
            ),
        ];
        for (err, expected) in cases {
            match Error::from(err) {
                Error::Client(ClientError::Api { status, message }) => {
                    assert_eq!(status, 0);
                    assert_eq!(message, expected);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn agent_runs_over_host_and_retries_once_on_invalid_json() {
        let host = MockLlm::scripted(vec![
            Ok(text("not json")),
            Ok(text(r#"{"chat_message":"parsed"}"#)),
        ]);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(HostLlmClient::new(host.clone()), "m"),
        );
        let out = block_on(agent.run_async(Some(BasicChatInputSchema {
            chat_message: "hi".to_string(),
        })))
        .unwrap();
        assert_eq!(out.chat_message, "parsed");
        let sent = host.requests();
        assert_eq!(sent.len(), 2, "one retry after the invalid reply");
        for request in &sent {
            assert!(request.tools.is_empty());
            assert!(request.output_schema.is_some());
        }
    }

    #[test]
    fn data_types_round_trip_through_serde() {
        let request = CompletionRequest {
            model: "m".into(),
            system: Some("s".into()),
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolCall(ToolCall {
                        id: "1".into(),
                        name: "n".into(),
                        arguments: "{}".into(),
                    }),
                    ContentBlock::ToolResult(ToolResult {
                        call_id: "1".into(),
                        content: "r".into(),
                        is_error: false,
                    }),
                ],
            }],
            tools: vec![],
            output_schema: None,
            max_tokens: Some(1),
            params: None,
        };
        let back: CompletionRequest =
            serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
        assert_eq!(back, request);
        assert!(
            !serde_json::to_string(&request)
                .unwrap()
                .contains("tool_choice")
        );
    }
}
