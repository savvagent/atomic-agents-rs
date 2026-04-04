use std::collections::HashMap;

use crate::chat_history::ChatHistory;
use crate::client::LlmClient;
use crate::system_prompt::SystemPromptGenerator;

/// Mode for extracting structured output from LLM responses.
#[derive(Clone, Debug, Default)]
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
    /// Conversation history (auto-created if None).
    pub history: Option<ChatHistory>,
    /// System prompt generator (auto-created if None).
    pub system_prompt_generator: Option<SystemPromptGenerator>,
    /// Role string for system messages (None disables system prompt).
    pub system_role: Option<String>,
    /// Extraction mode for structured output.
    pub mode: ExtractionMode,
    /// Maximum retry attempts for structured output extraction.
    pub max_retries: usize,
    /// Additional provider-specific parameters (temperature, max_tokens, etc.).
    pub model_api_parameters: HashMap<String, serde_json::Value>,
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
            mode: ExtractionMode::default(),
            max_retries: 3,
            model_api_parameters: HashMap::new(),
        }
    }

    pub fn with_history(mut self, history: ChatHistory) -> Self {
        self.history = Some(history);
        self
    }

    pub fn with_system_prompt(mut self, generator: SystemPromptGenerator) -> Self {
        self.system_prompt_generator = Some(generator);
        self
    }

    pub fn with_system_role(mut self, role: Option<String>) -> Self {
        self.system_role = role;
        self
    }

    pub fn with_mode(mut self, mode: ExtractionMode) -> Self {
        self.mode = mode;
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
}
