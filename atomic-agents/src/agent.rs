pub mod config;
pub mod hooks;
pub mod streaming;

use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Mutex;

use futures::stream::Stream;

use crate::chat_history::{ChatHistory, Role};
use crate::client::types::{
    ChatMessage, ChatRequest, FunctionDefinition, ResponseFormat, ToolDefinition,
};
use crate::client::LlmClient;
use crate::error::{Error, Result};
use crate::schema::IOSchema;
use crate::system_prompt::{ContextProvider, SystemPromptGenerator};

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
    history: Mutex<ChatHistory>,
    system_prompt_generator: SystemPromptGenerator,
    system_role: Option<String>,
    assistant_role: String,
    tool_result_role: String,
    initial_history: ChatHistory,
    mode: ExtractionMode,
    max_retries: usize,
    hooks: Mutex<HookRegistry>,
    extra_params: std::collections::HashMap<String, serde_json::Value>,
    _phantom: PhantomData<(I, O)>,
}

impl<I: IOSchema, O: IOSchema, C: LlmClient> AtomicAgent<I, O, C> {
    /// Creates a new agent from the given configuration.
    pub fn new(config: AgentConfig<C>) -> Self {
        let assistant_role = config.client.assistant_role().to_string();
        let tool_result_role = config.client.tool_result_role().to_string();

        let history = config.history.unwrap_or_default();
        let system_prompt_generator = config.system_prompt_generator.unwrap_or_default();

        Self {
            client: config.client,
            model: config.model,
            initial_history: history.copy(),
            history: Mutex::new(history),
            system_prompt_generator,
            system_role: config.system_role,
            assistant_role,
            tool_result_role,
            mode: config.mode,
            max_retries: config.max_retries,
            hooks: Mutex::new(HookRegistry::new()),
            extra_params: config.model_api_parameters,
            _phantom: PhantomData,
        }
    }

    /// Asynchronous single-turn execution.
    ///
    /// Sends the input to the LLM, extracts structured output, and returns
    /// the validated output schema. On parse failure, retries up to `max_retries`.
    pub async fn run_async(&self, user_input: Option<I>) -> Result<O> {
        if let Some(input) = &user_input {
            let mut history = self.history.lock().unwrap();
            history.initialize_turn();
            history.add_message(Role::User, input);
        }

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
                    history.add_message(
                        Role::Custom(self.assistant_role.clone()),
                        &output,
                    );
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
                            &response.content.chars().take(500).collect::<String>()
                        ),
                    });
                }
            }
        }
    }

    /// Asynchronous streaming execution.
    ///
    /// Returns a stream of `PartialOutput<O>` as chunks arrive. The final
    /// item in the stream will have `is_complete() == true`.
    pub async fn run_async_stream(
        &self,
        user_input: Option<I>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PartialOutput<O>>> + Send + '_>>> {
        if let Some(input) = &user_input {
            let mut history = self.history.lock().unwrap();
            history.initialize_turn();
            history.add_message(Role::User, input);
        }

        let messages = self.prepare_messages();
        let tools = self.build_tools_definition();
        let request = self.build_request(&messages, &tools, true);

        let chunk_stream = self.client.chat_completion_stream(request).await?;

        let stream = StreamAccumulator::<O>::new(chunk_stream);

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
            .register_context_provider(name, provider);
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
        *history = self.initial_history.copy();
    }

    /// Returns a copy of the current conversation history.
    pub fn history(&self) -> ChatHistory {
        self.history.lock().unwrap().copy()
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    // --- Private methods ---

    fn prepare_messages(&self) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // System message
        if let Some(ref role) = self.system_role {
            let mut prompt = self.system_prompt_generator.generate_prompt();

            // For JSON mode, append schema to system prompt
            if matches!(self.mode, ExtractionMode::Json | ExtractionMode::MdJson) {
                let schema = <O as IOSchema>::generate_schema();
                let schema_json =
                    serde_json::to_string_pretty(&schema).unwrap_or_default();
                prompt.push_str(&format!(
                    "\n\n# RESPONSE SCHEMA\nYou must respond with JSON matching this schema:\n```json\n{}\n```\n",
                    schema_json
                ));
            }

            messages.push(ChatMessage {
                role: role.clone(),
                content: serde_json::Value::String(prompt),
            });
        }

        // History messages
        let history = self.history.lock().unwrap();
        for msg in history.get_history() {
            let role = msg["role"].as_str().unwrap_or("user").to_string();
            let content = msg["content"].clone();

            // Remap "system" role in mid-conversation messages if needed
            let actual_role = if role == "system" && self.tool_result_role != "system" {
                self.tool_result_role.clone()
            } else {
                role
            };

            messages.push(ChatMessage {
                role: actual_role,
                content,
            });
        }

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

/// Accumulates stream chunks and yields partial outputs.
struct StreamAccumulator<O: IOSchema> {
    inner: Pin<Box<dyn Stream<Item = std::result::Result<crate::client::types::ChatChunk, Error>> + Send>>,
    accumulated: String,
    _phantom: PhantomData<O>,
}

impl<O: IOSchema> StreamAccumulator<O> {
    fn new(
        inner: Pin<Box<dyn Stream<Item = std::result::Result<crate::client::types::ChatChunk, Error>> + Send>>,
    ) -> Self {
        Self {
            inner,
            accumulated: String::new(),
            _phantom: PhantomData,
        }
    }
}

impl<O: IOSchema> Stream for StreamAccumulator<O> {
    type Item = Result<PartialOutput<O>>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = unsafe { self.get_unchecked_mut() };

        match Pin::new(&mut this.inner).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                this.accumulated.push_str(&chunk.delta);
                let partial = PartialOutput::new(this.accumulated.clone());
                std::task::Poll::Ready(Some(Ok(partial)))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                std::task::Poll::Ready(Some(Err(e)))
            }
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
