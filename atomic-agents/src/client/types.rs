use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A message in a chat completion request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

/// A function/tool definition for structured output via tool calling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

/// Function definition within a tool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Response format specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
}

/// A chat completion request to an LLM provider.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub response_format: Option<ResponseFormat>,
    pub stream: bool,
    /// Additional provider-specific parameters (temperature, max_tokens, etc.).
    pub extra: HashMap<String, serde_json::Value>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: None,
            response_format: None,
            stream: false,
            extra: HashMap::new(),
        }
    }
}

/// Token usage information from the API response.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// A complete (non-streaming) chat completion response.
#[derive(Clone, Debug)]
pub struct ChatResponse {
    /// The extracted content (either from message content or tool call arguments).
    pub content: String,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

impl Default for ChatResponse {
    fn default() -> Self {
        Self {
            content: String::new(),
            finish_reason: None,
            usage: None,
        }
    }
}

/// A single chunk from a streaming chat completion.
#[derive(Clone, Debug)]
pub struct ChatChunk {
    /// Delta content for this chunk.
    pub delta: String,
    pub finish_reason: Option<String>,
}
