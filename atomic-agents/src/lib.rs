//! # Atomic Agents
//!
//! A lightweight, modular framework for building AI agent pipelines with structured LLM outputs.
//!
//! ## Overview
//!
//! Atomic Agents provides:
//! - **Type-safe I/O schemas** via `#[derive(IOSchema)]`
//! - **Provider-agnostic LLM clients** (OpenAI, Anthropic, Google Gemini)
//! - **Structured output extraction** with automatic retries
//! - **Composable tools, resources, and prompts** via traits
//! - **Conversation history** with turn tracking and multimodal support
//! - **Hook system** for monitoring and error handling
//!
//! ## Quick Start
//!
//! ```ignore
//! use atomic_agents::prelude::*;
//!
//! let client = OpenAiClient::new("sk-...");
//! let config = AgentConfig::new(client, "gpt-4o");
//! let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);
//!
//! let response = agent.run(Some(BasicChatInputSchema {
//!     chat_message: "Hello!".to_string(),
//! })).unwrap();
//!
//! println!("{}", response.chat_message);
//! ```

pub mod agent;
pub mod chat_history;
pub mod client;
pub mod error;
pub mod mcp;
pub mod prompt;
pub mod resource;
pub mod schema;
pub mod system_prompt;
pub mod token_counter;
pub mod tool;

/// Re-exports of the most commonly used types.
pub mod prelude {
    pub use crate::agent::{
        AgentConfig, AtomicAgent, ExtractionMode, HookEvent, HookHandler, PartialOutput,
    };
    pub use crate::chat_history::{ChatHistory, ContentPart, Message, MessageContent, Role};
    pub use crate::client::LlmClient;
    pub use crate::client::types::{
        ChatChunk, ChatMessage, ChatRequest, ChatResponse, ToolDefinition, Usage,
    };
    pub use crate::error::{ClientError, Error, Result};
    pub use crate::prompt::Prompt;
    pub use crate::resource::Resource;
    pub use crate::schema::{BasicChatInputSchema, BasicChatOutputSchema, IOSchema};
    pub use crate::system_prompt::{ContextProvider, SystemPromptGenerator};
    pub use crate::token_counter::{TokenCountResult, TokenCounter, get_token_counter};
    pub use crate::tool::{ComponentConfig, Tool};

    #[cfg(feature = "openai")]
    pub use crate::client::openai::OpenAiClient;

    #[cfg(feature = "anthropic")]
    pub use crate::client::anthropic::AnthropicClient;

    #[cfg(feature = "google")]
    pub use crate::client::google::GoogleClient;

    pub use atomic_agents_macros::IOSchema;
}

// Re-export the derive macro at the crate root
pub use atomic_agents_macros::IOSchema;
