pub mod config;
pub mod hooks;
pub mod streaming;

use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Mutex;

use futures::stream::Stream;

use crate::chat_history::{BaseChatHistory, ChatHistory, ContentPart, Role};
use crate::client::types::{
    ChatMessage, ChatRequest, FunctionDefinition, ResponseFormat, ToolDefinition,
};
use crate::client::LlmClient;
use crate::error::{Error, Result};
use crate::schema::IOSchema;
use crate::system_prompt::{BaseSystemPromptGenerator, ContextProvider, SystemPromptGenerator};
use crate::token_counter::{TokenCountResult, TokenCounter};

pub use config::{AgentConfig, ExtractionMode};
pub use hooks::{HookEvent, HookHandler, HookRegistry};
pub use streaming::PartialOutput;

/// The core agent that orchestrates LLM interactions with structured I/O.
///
/// `AtomicAgent` takes structured input, sends it to an LLM with schema constraints,
/// and returns validated, typed output. It supports synchronous, asynchronous,
/// and streaming execution modes.
///
/// Mirrors Python's `AtomicAgent[InputSchema, OutputSchema]`.
///
/// A run that fails (client error, extraction failure, context budget exceeded)
/// removes the user turn it added, so the history never keeps a user message
/// without a response and a retry does not send the same request twice.
///
/// # Type Parameters
///
/// - `I`: Input schema type (implements `IOSchema`)
/// - `O`: Output schema type (implements `IOSchema`)
/// - `C`: LLM client type (implements `LlmClient`)
///
/// # Example
///
/// ```ignore
/// use atomic_agents::prelude::*;
///
/// let client = OpenAiClient::new("sk-...");
/// let config = AgentConfig::new(client, "gpt-4o");
/// let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);
/// let response = agent.run_async(Some(BasicChatInputSchema {
///     chat_message: "Hello!".to_string(),
/// })).await?;
/// ```
pub struct AtomicAgent<I: IOSchema, O: IOSchema, C: LlmClient> {
    client: C,
    model: String,
    history: Mutex<Box<dyn BaseChatHistory>>,
    system_prompt_generator: Box<dyn BaseSystemPromptGenerator>,
    system_role: Option<String>,
    assistant_role: String,
    tool_result_role: String,
    initial_history: Box<dyn BaseChatHistory>,
    mode: ExtractionMode,
    max_retries: usize,
    max_context_tokens: Option<usize>,
    hooks: Mutex<HookRegistry>,
    extra_params: std::collections::HashMap<String, serde_json::Value>,
    _phantom: PhantomData<(I, O)>,
}

impl<I: IOSchema, O: IOSchema, C: LlmClient> AtomicAgent<I, O, C> {
    /// Creates a new agent from the given configuration.
    ///
    /// The extraction mode is `config.mode` when set, otherwise the client's
    /// [`default_extraction_mode`](LlmClient::default_extraction_mode).
    pub fn new(config: AgentConfig<C>) -> Self {
        let assistant_role = config.client.assistant_role().to_string();
        let tool_result_role = config.client.tool_result_role().to_string();
        let mode = config
            .mode
            .unwrap_or_else(|| config.client.default_extraction_mode());

        let history: Box<dyn BaseChatHistory> = config
            .history
            .unwrap_or_else(|| Box::new(ChatHistory::default()));
        let system_prompt_generator = config
            .system_prompt_generator
            .unwrap_or_else(|| Box::new(SystemPromptGenerator::default()));

        Self {
            client: config.client,
            model: config.model,
            initial_history: history.boxed_copy(),
            history: Mutex::new(history),
            system_prompt_generator,
            system_role: config.system_role,
            assistant_role,
            tool_result_role,
            mode,
            max_retries: config.max_retries,
            max_context_tokens: config.max_context_tokens,
            hooks: Mutex::new(HookRegistry::new()),
            extra_params: config.model_api_parameters,
            _phantom: PhantomData,
        }
    }

    /// Asynchronous single-turn execution.
    ///
    /// Sends the input to the LLM, extracts structured output, and returns
    /// the validated output schema. On parse failure, retries up to `max_retries`.
    /// If the run fails, the user turn it added is removed from the history.
    pub async fn run_async(&self, user_input: Option<I>) -> Result<O> {
        let added_turn = self.begin_turn(user_input.as_ref());
        match self.execute().await {
            Ok(output) => Ok(output),
            Err(e) => {
                self.discard_turn(added_turn.as_deref());
                Err(e)
            }
        }
    }

    async fn execute(&self) -> Result<O> {
        self.trim_context()?;

        let mut messages = self.prepare_messages();
        let tools = self.build_tools_definition();

        let mut attempts = 0;
        loop {
            let request = self.build_request(&messages, &tools, false);

            self.dispatch_hook(
                &HookEvent::CompletionKwargs,
                &serde_json::to_value(&request.model).unwrap_or_default(),
            );

            let response = self.client.chat_completion(request).await?;

            self.dispatch_hook(
                &HookEvent::CompletionResponse,
                &serde_json::json!({"content": &response.content}),
            );

            match serde_json::from_str::<O>(&response.content) {
                Ok(output) => {
                    let mut history = self.history.lock().unwrap();
                    history.add_message(Role::Custom(self.assistant_role.clone()), &output);
                    return Ok(output);
                }
                Err(e) if attempts < self.max_retries => {
                    attempts += 1;

                    self.dispatch_hook(
                        &HookEvent::ParseError,
                        &serde_json::json!({
                            "error": e.to_string(),
                            "attempt": attempts,
                            "content": &response.content,
                        }),
                    );

                    if attempts == self.max_retries {
                        self.dispatch_hook(
                            &HookEvent::CompletionLastAttempt,
                            &serde_json::json!({"attempt": attempts}),
                        );
                    }

                    // Append error context for retry
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: serde_json::Value::String(response.content),
                    });
                    messages.push(ChatMessage {
                        role: "user".to_string(),
                        content: serde_json::Value::String(format!(
                            "Your response did not match the required JSON schema. Error: {}. \
                             Please try again and respond with valid JSON matching the schema.",
                            e
                        )),
                    });
                }
                Err(e) => {
                    self.dispatch_hook(
                        &HookEvent::CompletionError,
                        &serde_json::json!({"error": e.to_string()}),
                    );
                    return Err(Error::ExtractionFailed {
                        attempts,
                        message: format!(
                            "Failed to parse response as {}: {}. Last response: {}",
                            O::schema_title(),
                            e,
                            response.content.chars().take(500).collect::<String>()
                        ),
                    });
                }
            }
        }
    }

    /// Asynchronous streaming execution.
    ///
    /// Returns a stream of `PartialOutput<O>` as chunks arrive. The final
    /// item in the stream will have `is_complete() == true`. When the stream
    /// finishes with a valid output it is recorded in the history as the
    /// assistant message. If the request fails, the stream errors, the final
    /// output does not parse, or the stream is dropped before it finishes,
    /// the user turn this call added is removed from the history.
    pub async fn run_async_stream(
        &self,
        user_input: Option<I>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PartialOutput<O>>> + Send + '_>>> {
        let added_turn = self.begin_turn(user_input.as_ref());

        let started = async {
            self.trim_context()?;
            let messages = self.prepare_messages();
            let tools = self.build_tools_definition();
            let request = self.build_request(&messages, &tools, true);
            self.client.chat_completion_stream(request).await
        }
        .await;

        let chunk_stream = match started {
            Ok(stream) => stream,
            Err(e) => {
                self.discard_turn(added_turn.as_deref());
                return Err(e);
            }
        };

        let stream = StreamAccumulator::<O>::new(
            chunk_stream,
            &self.history,
            self.assistant_role.clone(),
            added_turn,
        );

        Ok(Box::pin(stream))
    }

    /// Synchronous single-turn execution (convenience wrapper).
    ///
    /// Blocks the current thread until the response is ready.
    /// Must be called from within a tokio runtime.
    pub fn run(&self, user_input: Option<I>) -> Result<O> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.run_async(user_input))
        })
    }

    /// Adds a tool/context result to the conversation history.
    ///
    /// Uses the provider-appropriate role (e.g., "system" for OpenAI, "user" for Gemini).
    pub fn add_tool_result<S: IOSchema>(&self, content: &S) {
        let mut history = self.history.lock().unwrap();
        history.add_message(Role::Custom(self.tool_result_role.clone()), content);
    }

    /// Registers a dynamic context provider.
    pub fn register_context_provider(
        &mut self,
        name: impl Into<String>,
        provider: Box<dyn ContextProvider>,
    ) {
        self.system_prompt_generator
            .register_context_provider(name.into(), provider);
    }

    /// Removes a context provider by name.
    pub fn unregister_context_provider(&mut self, name: &str) {
        self.system_prompt_generator
            .unregister_context_provider(name);
    }

    /// Registers a hook handler for an event.
    pub fn register_hook(&self, event: HookEvent, handler: HookHandler) {
        self.hooks.lock().unwrap().register(event, handler);
    }

    /// Removes a specific hook handler.
    pub fn unregister_hook(&self, event: &HookEvent, handler: &HookHandler) {
        self.hooks.lock().unwrap().unregister(event, handler);
    }

    /// Clears hook handlers for an event (or all events if None).
    pub fn clear_hooks(&self, event: Option<&HookEvent>) {
        self.hooks.lock().unwrap().clear(event);
    }

    /// Enables hook processing.
    pub fn enable_hooks(&self) {
        self.hooks.lock().unwrap().enable();
    }

    /// Disables hook processing.
    pub fn disable_hooks(&self) {
        self.hooks.lock().unwrap().disable();
    }

    /// Resets the conversation history to the initial state.
    pub fn reset_history(&self) {
        let mut history = self.history.lock().unwrap();
        *history = self.initial_history.boxed_copy();
    }

    /// Returns a copy of the current conversation history.
    pub fn history(&self) -> Box<dyn BaseChatHistory> {
        self.history.lock().unwrap().boxed_copy()
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the effective extraction mode.
    pub fn mode(&self) -> ExtractionMode {
        self.mode
    }

    /// Counts the tokens of the context that the next request would send
    /// (system prompt + history + tool definition), with a breakdown.
    ///
    /// Dispatches the local [`HookEvent::TokenCounted`] event.
    pub fn get_context_token_count(&self) -> Result<TokenCountResult> {
        let result = {
            let history = self.history.lock().unwrap();
            self.count_context_tokens(history.as_ref())?
        };
        self.dispatch_token_counted(&result);
        Ok(result)
    }

    // --- Private methods ---

    /// Starts a new turn for `user_input` and returns the turn ID it created.
    fn begin_turn(&self, user_input: Option<&I>) -> Option<String> {
        let input = user_input?;
        let mut history = self.history.lock().unwrap();
        let turn_id = history.initialize_turn();
        history.add_message(Role::User, input);
        Some(turn_id)
    }

    /// Removes the turn a failed run started, if it is still the current turn.
    fn discard_turn(&self, turn_id: Option<&str>) {
        discard_turn(&self.history, turn_id);
    }

    fn dispatch_token_counted(&self, result: &TokenCountResult) {
        self.dispatch_hook(
            &HookEvent::TokenCounted,
            &serde_json::json!({
                "total": result.total,
                "system_prompt": result.system_prompt,
                "history": result.history,
                "tools": result.tools,
                "model": result.model,
                "max_tokens": result.max_tokens,
                "utilization": result.utilization,
            }),
        );
    }

    /// Trims the oldest conversation turns until the context fits
    /// `max_context_tokens`. The current turn is always preserved.
    ///
    /// Fails with [`Error::TokenCount`] if the context still does not fit once
    /// only the current turn (plus system prompt and tools) is left, and with
    /// [`Error::Config`] for a zero budget.
    fn trim_context(&self) -> Result<()> {
        let Some(max) = self.max_context_tokens else {
            return Ok(());
        };
        config::validate_max_context_tokens(max)?;

        loop {
            let result = self.get_context_token_count()?;
            if result.total <= max {
                return Ok(());
            }

            let mut history = self.history.lock().unwrap();
            let current = history.current_turn_id().map(str::to_string);
            let oldest = history
                .turn_ids()
                .into_iter()
                .find(|id| Some(id) != current.as_ref());

            match oldest {
                Some(turn_id) => {
                    history.delete_turn(&turn_id);
                    tracing::warn!(
                        max_context_tokens = max,
                        turn_id = turn_id.as_str(),
                        total_tokens = result.total,
                        "Context exceeded max_context_tokens; trimmed oldest turn"
                    );
                }
                None => {
                    return Err(Error::TokenCount(format!(
                        "max_context_tokens ({max}) is smaller than the minimum required \
                         for a single turn ({} tokens). Increase max_context_tokens or \
                         reduce the system prompt size.",
                        result.total
                    )));
                }
            }
        }
    }

    fn count_context_tokens(&self, history: &dyn BaseChatHistory) -> Result<TokenCountResult> {
        let system_messages = self.build_system_messages();
        let history_messages: Vec<ChatMessage> = self
            .build_history_messages(history)
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: token_count_content(m.content),
            })
            .collect();
        let tools_json = self
            .build_tools_definition()
            .map(|tools| serde_json::to_string(&tools))
            .transpose()?;

        TokenCounter::count_context(
            &self.model,
            &system_messages,
            &history_messages,
            tools_json.as_deref(),
        )
    }

    fn build_system_messages(&self) -> Vec<ChatMessage> {
        let Some(ref role) = self.system_role else {
            return Vec::new();
        };
        let mut prompt = self.system_prompt_generator.generate_prompt();

        // For JSON mode, append schema to system prompt
        if matches!(self.mode, ExtractionMode::Json | ExtractionMode::MdJson) {
            let schema = <O as IOSchema>::generate_schema();
            let schema_json = serde_json::to_string_pretty(&schema).unwrap_or_default();
            prompt.push_str(&format!(
                "\n\n# RESPONSE SCHEMA\nYou must respond with JSON matching this schema:\n```json\n{}\n```\n",
                schema_json
            ));
        }

        vec![ChatMessage {
            role: role.clone(),
            content: serde_json::Value::String(prompt),
        }]
    }

    fn build_history_messages(&self, history: &dyn BaseChatHistory) -> Vec<ChatMessage> {
        history
            .get_history()
            .iter()
            .map(|msg| {
                let role = msg["role"].as_str().unwrap_or("user").to_string();
                let content = msg["content"].clone();

                // Remap "system" role in mid-conversation messages if needed
                let actual_role = if role == "system" && self.tool_result_role != "system" {
                    self.tool_result_role.clone()
                } else {
                    role
                };

                ChatMessage {
                    role: actual_role,
                    content,
                }
            })
            .collect()
    }

    fn prepare_messages(&self) -> Vec<ChatMessage> {
        let mut messages = self.build_system_messages();
        let history = self.history.lock().unwrap();
        messages.extend(self.build_history_messages(history.as_ref()));
        messages
    }

    fn build_tools_definition(&self) -> Option<Vec<ToolDefinition>> {
        match self.mode {
            ExtractionMode::Tools => {
                let schema = <O as IOSchema>::generate_schema();
                let mut params = serde_json::to_value(&schema).unwrap_or_default();

                // Remove $schema and title from parameters (not needed in tool def)
                if let serde_json::Value::Object(ref mut map) = params {
                    map.remove("$schema");
                }

                Some(vec![ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: O::schema_title().to_string(),
                        description: Some(O::schema_description().to_string()),
                        parameters: params,
                        strict: None,
                    },
                }])
            }
            ExtractionMode::Json | ExtractionMode::MdJson => None,
        }
    }

    fn build_request(
        &self,
        messages: &[ChatMessage],
        tools: &Option<Vec<ToolDefinition>>,
        stream: bool,
    ) -> ChatRequest {
        let mut request = ChatRequest::new(self.model.clone(), messages.to_vec());
        request.tools = tools.clone();
        request.stream = stream;
        request.extra = self.extra_params.clone();

        // Set response format for JSON mode
        if matches!(self.mode, ExtractionMode::Json) {
            request.response_format = Some(ResponseFormat {
                format_type: "json_object".to_string(),
                json_schema: None,
            });
        }

        request
    }

    fn dispatch_hook(&self, event: &HookEvent, data: &serde_json::Value) {
        self.hooks.lock().unwrap().dispatch(event, data);
    }
}

fn discard_turn(history: &Mutex<Box<dyn BaseChatHistory>>, turn_id: Option<&str>) {
    let Some(turn_id) = turn_id else {
        return;
    };
    if let Ok(mut history) = history.lock()
        && history.current_turn_id() == Some(turn_id)
    {
        history.delete_turn(turn_id);
    }
}

/// Flattens message content for token counting. Multimodal parts other than
/// text become short placeholders, since counting raw base64 payloads or URLs
/// would not reflect what the provider charges for them.
fn token_count_content(content: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Array(parts) = &content else {
        return content;
    };
    let text = parts
        .iter()
        .map(|part| match serde_json::from_value::<ContentPart>(part.clone()) {
            Ok(ContentPart::Text { text }) => text,
            Ok(ContentPart::ImageUrl { .. } | ContentPart::ImageBase64 { .. }) => {
                "[image content]".to_string()
            }
            Ok(ContentPart::AudioBase64 { .. }) => "[audio content]".to_string(),
            Ok(ContentPart::VideoUrl { .. }) => "[video content]".to_string(),
            Err(_) => {
                let kind = part.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                format!("[{kind} content]")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    serde_json::Value::String(text)
}

type ChunkStream =
    Pin<Box<dyn Stream<Item = std::result::Result<crate::client::types::ChatChunk, Error>> + Send>>;

/// Accumulates stream chunks and yields partial outputs.
///
/// Records the final output in the history on success and discards the user
/// turn on any failure (including being dropped before the stream finishes).
struct StreamAccumulator<'a, O: IOSchema> {
    inner: ChunkStream,
    accumulated: String,
    history: &'a Mutex<Box<dyn BaseChatHistory>>,
    assistant_role: String,
    turn_id: Option<String>,
    finished: bool,
    _phantom: PhantomData<fn() -> O>,
}

impl<'a, O: IOSchema> StreamAccumulator<'a, O> {
    fn new(
        inner: ChunkStream,
        history: &'a Mutex<Box<dyn BaseChatHistory>>,
        assistant_role: String,
        turn_id: Option<String>,
    ) -> Self {
        Self {
            inner,
            accumulated: String::new(),
            history,
            assistant_role,
            turn_id,
            finished: false,
            _phantom: PhantomData,
        }
    }

    fn fail(&mut self) {
        self.finished = true;
        discard_turn(self.history, self.turn_id.as_deref());
    }
}

impl<O: IOSchema> Drop for StreamAccumulator<'_, O> {
    fn drop(&mut self) {
        if !self.finished {
            self.fail();
        }
    }
}

impl<O: IOSchema> Stream for StreamAccumulator<'_, O> {
    type Item = Result<PartialOutput<O>>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;

        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }

        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.accumulated.push_str(&chunk.delta);
                let partial = PartialOutput::new(this.accumulated.clone());
                Poll::Ready(Some(Ok(partial)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.fail();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => match serde_json::from_str::<O>(&this.accumulated) {
                Ok(output) => {
                    this.finished = true;
                    if let Ok(mut history) = this.history.lock() {
                        history.add_message(Role::Custom(this.assistant_role.clone()), &output);
                    }
                    Poll::Ready(None)
                }
                Err(e) => {
                    this.fail();
                    Poll::Ready(Some(Err(Error::ExtractionFailed {
                        attempts: 0,
                        message: format!(
                            "Failed to parse streamed response as {}: {}. Last response: {}",
                            O::schema_title(),
                            e,
                            this.accumulated.chars().take(500).collect::<String>()
                        ),
                    })))
                }
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_history::MessageContent;
    use crate::client::mock::MockClient;
    use crate::schema::{BasicChatInputSchema, BasicChatOutputSchema};

    #[tokio::test]
    async fn test_agent_run_async() {
        let response_json = serde_json::json!({
            "chat_message": "Hello! How can I help you?"
        });
        let client = MockClient::new(response_json.to_string());
        let config = AgentConfig::new(client, "test-model");
        let agent =
            AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

        let result = agent
            .run_async(Some(BasicChatInputSchema {
                chat_message: "Hi".to_string(),
            }))
            .await
            .unwrap();

        assert_eq!(result.chat_message, "Hello! How can I help you?");
    }

    #[tokio::test]
    async fn test_agent_history_persistence() {
        let response_json = serde_json::json!({
            "chat_message": "Response 1"
        });
        let client = MockClient::new(response_json.to_string());
        let config = AgentConfig::new(client, "test-model");
        let agent =
            AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

        agent
            .run_async(Some(BasicChatInputSchema {
                chat_message: "Message 1".to_string(),
            }))
            .await
            .unwrap();

        let history = agent.history();
        // Should have 2 messages: user input + assistant response
        assert_eq!(history.message_count(), 2);
    }

    #[tokio::test]
    async fn test_agent_reset_history() {
        let response_json = serde_json::json!({
            "chat_message": "Response"
        });
        let client = MockClient::new(response_json.to_string());
        let config = AgentConfig::new(client, "test-model");
        let agent =
            AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

        agent
            .run_async(Some(BasicChatInputSchema {
                chat_message: "Hello".to_string(),
            }))
            .await
            .unwrap();

        assert!(agent.history().message_count() > 0);
        agent.reset_history();
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_agent_extraction_failure_retries() {
        let client = MockClient::new("not valid json");
        let config = AgentConfig::new(client, "test-model").with_max_retries(1);
        let agent =
            AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

        let result = agent
            .run_async(Some(BasicChatInputSchema {
                chat_message: "test".to_string(),
            }))
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            Error::ExtractionFailed { attempts, .. } => {
                assert_eq!(attempts, 1);
            }
            other => panic!("Expected ExtractionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_agent_hooks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let response_json = serde_json::json!({"chat_message": "hi"});
        let client = MockClient::new(response_json.to_string());
        let config = AgentConfig::new(client, "test-model");
        let agent =
            AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let handler: HookHandler = Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        agent.register_hook(HookEvent::CompletionResponse, handler);

        agent
            .run_async(Some(BasicChatInputSchema {
                chat_message: "test".to_string(),
            }))
            .await
            .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // --- Ported upstream behaviour ---

    use futures::StreamExt;

    fn input(text: &str) -> BasicChatInputSchema {
        BasicChatInputSchema {
            chat_message: text.to_string(),
        }
    }

    fn ok_agent() -> AtomicAgent<BasicChatInputSchema, BasicChatOutputSchema, MockClient> {
        let client = MockClient::new(r#"{"chat_message": "ok"}"#);
        AtomicAgent::new(AgentConfig::new(client, "gpt-4o"))
    }

    /// An output schema whose JSON field name differs from the Rust field name.
    #[derive(
        Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema, PartialEq,
    )]
    struct AliasedOutput {
        /// The reply text.
        #[serde(rename = "reply", alias = "answer")]
        text: String,
    }

    impl IOSchema for AliasedOutput {
        fn schema_title() -> &'static str {
            "AliasedOutput"
        }
        fn schema_description() -> &'static str {
            "Output with an aliased field."
        }
    }

    // #475: reject nonpositive context token budgets

    #[test]
    fn test_zero_context_budget_rejected_by_builder() {
        let result = AgentConfig::new(MockClient::new("{}"), "gpt-4o").with_max_context_tokens(0);
        assert!(matches!(result, Err(Error::Config(_))));
        let config = AgentConfig::new(MockClient::new("{}"), "gpt-4o")
            .with_max_context_tokens(1)
            .unwrap();
        assert_eq!(config.max_context_tokens, Some(1));
    }

    #[tokio::test]
    async fn test_zero_context_budget_rejected_at_run() {
        let mut config = AgentConfig::new(MockClient::new(r#"{"chat_message": "ok"}"#), "gpt-4o");
        config.max_context_tokens = Some(0);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

        let err = agent.run_async(Some(input("hi"))).await.unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(agent.client.requests.lock().unwrap().is_empty());
        assert_eq!(agent.history().message_count(), 0);
    }

    // #301 and context budget: trim oldest turns, preserve the current turn

    #[tokio::test]
    async fn test_context_trim_drops_oldest_turns_and_keeps_current() {
        let client = MockClient::new(r#"{"chat_message": "ok"}"#);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        for i in 0..3 {
            agent
                .run_async(Some(input(&format!("older message {i}"))))
                .await
                .unwrap();
        }
        assert_eq!(agent.history().message_count(), 6);

        // Budget that fits the prompt, tools and roughly one turn, not four.
        let one_turn = {
            let fresh = ok_agent();
            fresh.run_async(Some(input("older message 0"))).await.unwrap();
            fresh.get_context_token_count().unwrap().total
        };
        let mut agent = agent;
        agent.max_context_tokens = Some(one_turn + 5);

        agent.run_async(Some(input("the newest message"))).await.unwrap();

        let sent = agent.client.requests.lock().unwrap().last().unwrap().clone();
        let sent_text = serde_json::to_string(&sent.messages).unwrap();
        assert!(sent_text.contains("the newest message"));
        assert!(!sent_text.contains("older message 0"));
        assert!(agent.history().message_count() < 8);
    }

    #[tokio::test]
    async fn test_context_overflow_with_only_current_turn_errors_and_discards_it() {
        let client = MockClient::new(r#"{"chat_message": "ok"}"#);
        let mut config = AgentConfig::new(client, "gpt-4o");
        config.max_context_tokens = Some(1);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

        let err = agent.run_async(Some(input("hello"))).await.unwrap_err();
        assert!(matches!(err, Error::TokenCount(_)));
        // Nothing was sent, and the current turn is gone rather than half-kept.
        assert!(agent.client.requests.lock().unwrap().is_empty());
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_context_trim_preserves_current_turn_even_if_oldest() {
        // Only one turn exists and it is the current one: it must never be trimmed.
        let client = MockClient::new(r#"{"chat_message": "ok"}"#);
        let mut config = AgentConfig::new(client, "gpt-4o");
        config.max_context_tokens = Some(100_000);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);
        agent.run_async(Some(input("only turn"))).await.unwrap();
        assert_eq!(agent.history().message_count(), 2);
    }

    // #481: drop the user turn when a run fails

    #[tokio::test]
    async fn test_failed_run_discards_user_turn() {
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(MockClient::failing(), "gpt-4o"),
        );
        assert!(agent.run_async(Some(input("hi"))).await.is_err());
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_failed_run_keeps_earlier_turns() {
        let client = MockClient::new(r#"{"chat_message": "ok"}"#);
        let mut agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        agent.run_async(Some(input("first"))).await.unwrap();
        assert_eq!(agent.history().message_count(), 2);

        agent.client.fail = true;
        assert!(agent.run_async(Some(input("second"))).await.is_err());
        assert_eq!(agent.history().message_count(), 2);
    }

    #[tokio::test]
    async fn test_extraction_failure_discards_user_turn() {
        let config = AgentConfig::new(MockClient::new("nope"), "gpt-4o").with_max_retries(0);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);
        assert!(agent.run_async(Some(input("hi"))).await.is_err());
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_run_without_input_failure_does_not_touch_history() {
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(MockClient::failing(), "gpt-4o"),
        );
        agent.add_tool_result(&input("context"));
        assert!(agent.run_async(None).await.is_err());
        assert_eq!(agent.history().message_count(), 1);
    }

    #[tokio::test]
    async fn test_stream_request_failure_discards_user_turn() {
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(MockClient::failing(), "gpt-4o"),
        );
        assert!(agent.run_async_stream(Some(input("hi"))).await.is_err());
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_stream_error_midway_discards_user_turn() {
        let mut client = MockClient::new("").with_stream_chunks(&[r#"{"chat_"#]);
        client.stream_error = true;
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        let mut stream = agent.run_async_stream(Some(input("hi"))).await.unwrap();
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            saw_error |= item.is_err();
        }
        drop(stream);
        assert!(saw_error);
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_stream_dropped_early_discards_user_turn() {
        let client = MockClient::new("").with_stream_chunks(&[r#"{"chat_"#, r#"message": "x"}"#]);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        let mut stream = agent.run_async_stream(Some(input("hi"))).await.unwrap();
        stream.next().await.unwrap().unwrap();
        drop(stream);
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_stream_unparseable_final_output_is_an_error_and_discards_turn() {
        let client = MockClient::new("").with_stream_chunks(&["not json"]);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        let mut stream = agent.run_async_stream(Some(input("hi"))).await.unwrap();
        let mut last = None;
        while let Some(item) = stream.next().await {
            last = Some(item);
        }
        drop(stream);
        assert!(matches!(last, Some(Err(Error::ExtractionFailed { .. }))));
        assert_eq!(agent.history().message_count(), 0);
    }

    #[tokio::test]
    async fn test_stream_success_records_assistant_message() {
        let client = MockClient::new("").with_stream_chunks(&[r#"{"chat_"#, r#"message": "hey"}"#]);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        let mut stream = agent.run_async_stream(Some(input("hi"))).await.unwrap();
        let mut last_complete = false;
        while let Some(item) = stream.next().await {
            last_complete = item.unwrap().is_complete();
        }
        drop(stream);
        assert!(last_complete);
        let history = agent.history().get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[1]["role"], "assistant");
    }

    // #468: accept JSON whitespace before streamed objects

    #[tokio::test]
    async fn test_stream_accepts_leading_whitespace() {
        let client = MockClient::new("").with_stream_chunks(&[
            "  \n",
            "\t{\"chat_message\"",
            ": \"hello\"}",
        ]);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        let mut stream = agent.run_async_stream(Some(input("hi"))).await.unwrap();
        let mut final_output = None;
        while let Some(item) = stream.next().await {
            let partial = item.unwrap();
            if partial.is_complete() {
                final_output = partial.parsed;
            }
        }
        assert_eq!(final_output.unwrap().chat_message, "hello");
    }

    // #314: streaming works with output schemas that use field aliases

    #[tokio::test]
    async fn test_stream_with_aliased_output_fields() {
        let client = MockClient::new("")
            .with_stream_chunks(&[r#"{"repl"#, r#"y": "aliased"}"#]);
        let agent = AtomicAgent::<BasicChatInputSchema, AliasedOutput, _>::new(AgentConfig::new(
            client, "gpt-4o",
        ));
        let mut stream = agent.run_async_stream(Some(input("hi"))).await.unwrap();
        let mut parsed = None;
        while let Some(item) = stream.next().await {
            parsed = item.unwrap().parsed.or(parsed);
        }
        drop(stream);
        assert_eq!(parsed.unwrap().text, "aliased");
        // Re-reading the recorded assistant message also round-trips.
        assert_eq!(agent.history().message_count(), 2);
    }

    // #293: mode derived from the client

    #[test]
    fn test_mode_defaults_to_client_mode() {
        let client = MockClient::new("{}").with_mode(ExtractionMode::Json);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "m"),
        );
        assert_eq!(agent.mode(), ExtractionMode::Json);
        assert!(agent.build_tools_definition().is_none());
    }

    #[test]
    fn test_explicit_mode_overrides_client_mode() {
        let client = MockClient::new("{}").with_mode(ExtractionMode::Json);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "m").with_mode(ExtractionMode::Tools),
        );
        assert_eq!(agent.mode(), ExtractionMode::Tools);
        assert!(agent.build_tools_definition().is_some());
    }

    #[test]
    fn test_mode_without_client_override_is_tools() {
        let agent = ok_agent();
        assert_eq!(agent.mode(), ExtractionMode::Tools);
    }

    // BaseChatHistory / BaseSystemPromptGenerator abstractions

    struct FixedPrompt;
    impl BaseSystemPromptGenerator for FixedPrompt {
        fn generate_prompt(&self) -> String {
            "FIXED PROMPT".to_string()
        }
    }

    #[tokio::test]
    async fn test_custom_system_prompt_generator() {
        let config =
            AgentConfig::new(MockClient::new(r#"{"chat_message": "ok"}"#), "m").with_system_prompt(FixedPrompt);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);
        agent.run_async(Some(input("hi"))).await.unwrap();
        let sent = agent.client.requests.lock().unwrap()[0].clone();
        assert_eq!(sent.messages[0].content, "FIXED PROMPT");
    }

    #[tokio::test]
    async fn test_reset_history_restores_initial_custom_history() {
        let mut seed = ChatHistory::new(None);
        seed.add_raw_message(Role::User, MessageContent::text("seed"));
        let config =
            AgentConfig::new(MockClient::new(r#"{"chat_message": "ok"}"#), "m").with_history(seed);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);
        agent.run_async(Some(input("hi"))).await.unwrap();
        assert_eq!(agent.history().message_count(), 3);
        agent.reset_history();
        assert_eq!(agent.history().message_count(), 1);
    }

    // Token counting on the agent

    #[tokio::test]
    async fn test_agent_context_token_count_is_additive_and_fires_hook() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let agent = ok_agent();
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = fired.clone();
        agent.register_hook(
            HookEvent::TokenCounted,
            Arc::new(move |_| {
                fired_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );
        agent.run_async(Some(input("count me"))).await.unwrap();

        let result = agent.get_context_token_count().unwrap();
        assert_eq!(
            result.total,
            result.system_prompt + result.history + result.tools
        );
        assert!(result.tools > 0, "Tools mode counts the schema");
        assert!(result.history > 0);
        assert!(fired.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_token_hook_is_local_and_respects_disable() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let agent = ok_agent();
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = fired.clone();
        agent.register_hook(
            HookEvent::TokenCounted,
            Arc::new(move |_| {
                fired_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );
        agent.disable_hooks();
        agent.get_context_token_count().unwrap();
        assert_eq!(fired.load(Ordering::SeqCst), 0);
        agent.enable_hooks();
        agent.get_context_token_count().unwrap();
        assert_eq!(fired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_json_mode_counts_schema_in_system_prompt_not_tools() {
        let client = MockClient::new("{}").with_mode(ExtractionMode::Json);
        let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(
            AgentConfig::new(client, "gpt-4o"),
        );
        let result = agent.get_context_token_count().unwrap();
        assert_eq!(result.tools, 0);
        assert!(result.system_prompt > 0);
    }

    #[test]
    fn test_token_count_content_uses_placeholders_for_media() {
        let content = serde_json::to_value(vec![
            ContentPart::Text {
                text: "look".to_string(),
            },
            ContentPart::ImageBase64 {
                data: "A".repeat(10_000),
                media_type: "image/png".to_string(),
            },
            ContentPart::VideoUrl {
                video_url: crate::chat_history::VideoUrlContent::new("https://e.com/v.mp4"),
            },
        ])
        .unwrap();
        let flat = token_count_content(content);
        let text = flat.as_str().unwrap();
        assert!(text.contains("look"));
        assert!(text.contains("[image content]"));
        assert!(text.contains("[video content]"));
        assert!(text.len() < 200);
    }
}
