use crate::client::types::ChatMessage;
use crate::error::Error;

/// Result of a token count operation with breakdown.
#[derive(Clone, Debug)]
pub struct TokenCountResult {
    /// Total tokens (system + history + tools).
    pub total: usize,
    /// Tokens used by the system prompt.
    pub system_prompt: usize,
    /// Tokens used by conversation history.
    pub history: usize,
    /// Tokens used by tool/schema definitions.
    pub tools: usize,
    /// The model used for counting.
    pub model: String,
    /// Maximum context window for the model (if known).
    pub max_tokens: Option<usize>,
    /// Context utilization as a fraction (total / max_tokens).
    pub utilization: Option<f64>,
}

/// Provider-agnostic token counter.
///
/// Uses `tiktoken-rs` for OpenAI models and approximate counting for others.
pub struct TokenCounter;

impl TokenCounter {
    /// Counts tokens in a list of messages.
    pub fn count_messages(model: &str, messages: &[ChatMessage]) -> Result<usize, Error> {
        let mut total = 0;
        for msg in messages {
            // Per-message overhead (role tokens, separators)
            total += 4; // <im_start>{role}\n ... <im_end>\n

            // Role tokens
            total += 1;

            // Content tokens
            let content_str = match &msg.content {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            total += Self::count_text(model, &content_str)?;
        }
        // Every reply is primed with <im_start>assistant
        total += 2;
        Ok(total)
    }

    /// Counts tokens in a text string.
    pub fn count_text(model: &str, text: &str) -> Result<usize, Error> {
        // Try tiktoken for OpenAI models
        let bpe = Self::get_tokenizer(model);
        match bpe {
            Some(bpe) => Ok(bpe.encode_with_special_tokens(text).len()),
            None => {
                // Approximate: ~4 characters per token for non-OpenAI models
                Ok((text.len() + 3) / 4)
            }
        }
    }

    /// Returns the maximum context window for a model (if known).
    pub fn max_tokens(model: &str) -> Option<usize> {
        // Common model context windows
        match model {
            m if m.contains("gpt-4o") => Some(128_000),
            m if m.contains("gpt-4-turbo") => Some(128_000),
            m if m.contains("gpt-4") => Some(8_192),
            m if m.contains("gpt-3.5-turbo") => Some(16_385),
            m if m.contains("claude-3") || m.contains("claude-sonnet") || m.contains("claude-opus") || m.contains("claude-haiku") => Some(200_000),
            m if m.contains("gemini-2") => Some(1_048_576),
            m if m.contains("gemini-1.5") => Some(1_048_576),
            m if m.contains("gemini-1.0") => Some(32_768),
            _ => None,
        }
    }

    /// Counts tokens across all context components with breakdown.
    ///
    /// The breakdown is additive: `system_prompt + history + tools == total`.
    /// Request framing is counted once, over the complete message list, and is
    /// attributed to the system prompt when system messages are present or to
    /// history otherwise. Counting the system prompt and history independently
    /// would charge that framing twice.
    pub fn count_context(
        model: &str,
        system_messages: &[ChatMessage],
        history_messages: &[ChatMessage],
        tools_json: Option<&str>,
    ) -> Result<TokenCountResult, Error> {
        let system_prompt = if system_messages.is_empty() {
            0
        } else {
            Self::count_messages(model, system_messages)?
        };

        let history = if history_messages.is_empty() {
            0
        } else {
            let all: Vec<ChatMessage> = system_messages
                .iter()
                .chain(history_messages)
                .cloned()
                .collect();
            Self::count_messages(model, &all)?.saturating_sub(system_prompt)
        };

        let tools = match tools_json {
            Some(json) => Self::count_text(model, json)?,
            None => 0,
        };

        let total = system_prompt + history + tools;
        let max_tokens = Self::max_tokens(model);
        let utilization = max_tokens.map(|max| total as f64 / max as f64);

        Ok(TokenCountResult {
            total,
            system_prompt,
            history,
            tools,
            model: model.to_string(),
            max_tokens,
            utilization,
        })
    }

    fn get_tokenizer(model: &str) -> Option<tiktoken_rs::CoreBPE> {
        // Try to get the encoding for the model
        tiktoken_rs::get_bpe_from_model(model).ok()
    }
}

/// Singleton accessor for convenience (matches Python's `get_token_counter()`).
pub fn get_token_counter() -> TokenCounter {
    TokenCounter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_text_openai_model() {
        let count = TokenCounter::count_text("gpt-4o", "Hello, world!").unwrap();
        assert!(count > 0);
        assert!(count < 10); // "Hello, world!" should be a few tokens
    }

    #[test]
    fn test_count_text_unknown_model() {
        let count = TokenCounter::count_text("unknown-model", "Hello, world!").unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_count_messages() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String("Hello".to_string()),
        }];
        let count = TokenCounter::count_messages("gpt-4o", &messages).unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_max_tokens() {
        assert_eq!(TokenCounter::max_tokens("gpt-4o"), Some(128_000));
        assert_eq!(TokenCounter::max_tokens("claude-3-opus"), Some(200_000));
        assert_eq!(TokenCounter::max_tokens("unknown"), None);
    }

    #[test]
    fn test_count_context() {
        let system = vec![ChatMessage {
            role: "system".to_string(),
            content: serde_json::Value::String("You are helpful.".to_string()),
        }];
        let history = vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String("Hi".to_string()),
        }];

        let result = TokenCounter::count_context("gpt-4o", &system, &history, None).unwrap();
        assert!(result.total > 0);
        assert!(result.system_prompt > 0);
        assert!(result.history > 0);
        assert_eq!(result.tools, 0);
        assert!(result.utilization.is_some());
    }

    #[test]
    fn test_count_text_has_no_chat_framing() {
        let text = "Hello, world!";
        let plain = TokenCounter::count_text("gpt-4o", text).unwrap();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String(text.to_string()),
        }];
        let framed = TokenCounter::count_messages("gpt-4o", &messages).unwrap();
        assert!(plain < framed);
    }

    #[test]
    fn test_count_context_breakdown_is_additive_without_double_framing() {
        let model = "gpt-4o";
        let system = vec![ChatMessage {
            role: "system".to_string(),
            content: serde_json::Value::String("You are helpful.".to_string()),
        }];
        let history = vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String("Hi there".to_string()),
        }];
        let all: Vec<ChatMessage> = system.iter().chain(&history).cloned().collect();

        let result =
            TokenCounter::count_context(model, &system, &history, Some("{\"type\":\"object\"}"))
                .unwrap();
        let tools = TokenCounter::count_text(model, "{\"type\":\"object\"}").unwrap();

        assert_eq!(
            result.total,
            result.system_prompt + result.history + result.tools
        );
        assert_eq!(result.tools, tools);
        // Framing is charged once for the whole request.
        assert_eq!(
            result.system_prompt + result.history,
            TokenCounter::count_messages(model, &all).unwrap()
        );
    }

    #[test]
    fn test_count_context_history_only_carries_framing() {
        let history = vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String("Hi".to_string()),
        }];
        let result = TokenCounter::count_context("gpt-4o", &[], &history, None).unwrap();
        assert_eq!(result.system_prompt, 0);
        assert_eq!(
            result.history,
            TokenCounter::count_messages("gpt-4o", &history).unwrap()
        );
        assert_eq!(result.total, result.history);
    }

    #[test]
    fn test_count_context_empty_is_zero() {
        let result = TokenCounter::count_context("gpt-4o", &[], &[], None).unwrap();
        assert_eq!(result.total, 0);
    }
}
