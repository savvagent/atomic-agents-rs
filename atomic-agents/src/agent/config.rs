use std::collections::HashMap;

use crate::chat_history::BaseChatHistory;
use crate::client::LlmClient;
use crate::error::{Error, Result};
use crate::system_prompt::BaseSystemPromptGenerator;

/// Mode for extracting structured output from LLM responses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExtractionMode {
    /// Use function/tool calling to extract structured output (most reliable).
    #[default]
    Tools,
    /// Use JSON mode with schema injected into the system prompt.
    Json,
    /// Use Markdown JSON mode (schema in system prompt, response in ```json blocks).
    MdJson,
}

/// Configuration for an `AtomicAgent`.
///
/// Mirrors Python's `AgentConfig`.
pub struct AgentConfig<C: LlmClient> {
    /// The LLM client to use for completions.
    pub client: C,
    /// The model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
    pub model: String,
    /// Conversation history (an in-memory `ChatHistory` if None).
    pub history: Option<Box<dyn BaseChatHistory>>,
    /// System prompt generator (a default `SystemPromptGenerator` if None).
    pub system_prompt_generator: Option<Box<dyn BaseSystemPromptGenerator>>,
    /// Role string for system messages (None disables system prompt).
    pub system_role: Option<String>,
    /// Extraction mode for structured output. When `None`, the client's
    /// [`default_extraction_mode`](LlmClient::default_extraction_mode) is used.
    pub mode: Option<ExtractionMode>,
    /// Maximum retry attempts for structured output extraction.
    pub max_retries: usize,
    /// Additional provider-specific parameters (temperature, max_tokens, etc.).
    pub model_api_parameters: HashMap<String, serde_json::Value>,
    /// Maximum tokens for the full context (system prompt + history + tools).
    /// `None` disables trimming. Must be positive; a zero budget makes every
    /// run fail with [`Error::Config`].
    pub max_context_tokens: Option<usize>,
    /// Overrides the JSON Schema sent to the LLM for the output type. Parsing
    /// still deserialises into `O`, so the override must describe a shape `O`
    /// accepts. Used for schemas only known at runtime, e.g. MCP orchestrators.
    pub output_schema: Option<serde_json::Value>,
}

impl<C: LlmClient> AgentConfig<C> {
    /// Creates a new agent config with the given client and model.
    pub fn new(client: C, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            history: None,
            system_prompt_generator: None,
            system_role: Some("system".to_string()),
            mode: None,
            max_retries: 3,
            model_api_parameters: HashMap::new(),
            max_context_tokens: None,
            output_schema: None,
        }
    }

    pub fn with_history(mut self, history: impl BaseChatHistory + 'static) -> Self {
        self.history = Some(Box::new(history));
        self
    }

    pub fn with_system_prompt(mut self, generator: impl BaseSystemPromptGenerator + 'static) -> Self {
        self.system_prompt_generator = Some(Box::new(generator));
        self
    }

    pub fn with_system_role(mut self, role: Option<String>) -> Self {
        self.system_role = role;
        self
    }

    /// Sets the extraction mode explicitly, overriding the client's default.
    pub fn with_mode(mut self, mode: ExtractionMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.model_api_parameters.insert(key.into(), value);
        self
    }

    /// Overrides the output JSON Schema sent to the LLM. See
    /// [`output_schema`](Self::output_schema).
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Sets the maximum context size in tokens; the oldest turns are trimmed
    /// when a run would exceed it.
    ///
    /// Returns [`Error::Config`] if `max_context_tokens` is zero.
    ///
    /// ```ignore
    /// let config = AgentConfig::new(client, "gpt-4o").with_max_context_tokens(8_000)?;
    /// ```
    pub fn with_max_context_tokens(mut self, max_context_tokens: usize) -> Result<Self> {
        validate_max_context_tokens(max_context_tokens)?;
        self.max_context_tokens = Some(max_context_tokens);
        Ok(self)
    }
}

pub(crate) fn validate_max_context_tokens(value: usize) -> Result<()> {
    if value == 0 {
        return Err(Error::Config(
            "max_context_tokens must be a positive number of tokens".to_string(),
        ));
    }
    Ok(())
}
