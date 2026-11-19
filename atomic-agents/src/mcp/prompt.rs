//! [`McpPrompt`]: a discovered MCP prompt exposed through the framework's
//! [`Prompt`] trait.

use rmcp::model::{ContentBlock, GetPromptResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::client::McpClient;
use super::definitions::McpPromptDefinition;
use super::tool::{McpToolInput, block_on_multi_thread};
use crate::error::{Error, Result};
use crate::prompt::Prompt;
use crate::schema::IOSchema;
use crate::tool::ComponentConfig;

/// One message of a rendered MCP prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct McpPromptMessage {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// The message's content block as the server sent it (text, image,
    /// audio, embedded resource or resource link), as JSON.
    pub content: Value,
}

/// Result of rendering an MCP prompt (`prompts/get`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct McpPromptOutput {
    /// The server's description of the rendered prompt, if any.
    pub description: Option<String>,
    /// Every message, in order.
    pub messages: Vec<McpPromptMessage>,
    /// The text of every text block, joined by a blank line. Non-text blocks
    /// contribute nothing here; read them from `messages`.
    pub content: String,
}

impl McpPromptOutput {
    pub(crate) fn from_result(result: GetPromptResult) -> Result<Self> {
        let mut texts = Vec::new();
        let mut messages = Vec::with_capacity(result.messages.len());
        for message in result.messages {
            if let ContentBlock::Text(t) = &message.content {
                texts.push(t.text.clone());
            }
            let role = match serde_json::to_value(&message.role) {
                Ok(Value::String(role)) => role,
                _ => {
                    return Err(Error::Mcp(format!(
                        "prompts/get returned a message with an unsupported role: {:?}",
                        message.role
                    )));
                }
            };
            let content = serde_json::to_value(&message.content).map_err(|e| {
                Error::Mcp(format!("prompts/get returned unserializable content: {e}"))
            })?;
            messages.push(McpPromptMessage { role, content });
        }
        Ok(Self {
            description: result.description,
            messages,
            content: texts.join("\n\n"),
        })
    }
}

impl IOSchema for McpPromptOutput {
    fn schema_title() -> &'static str {
        "McpPromptOutput"
    }
    fn schema_description() -> &'static str {
        "A rendered MCP prompt."
    }
}

/// An MCP prompt bound to the session it was discovered on.
///
/// Implements [`Prompt`] with the prompt arguments as input.
/// [`Prompt::generate`] is synchronous, so it blocks the calling thread and
/// requires a multi-threaded tokio runtime; in async code use
/// [`McpPrompt::get`].
#[derive(Clone, Debug)]
pub struct McpPrompt {
    definition: McpPromptDefinition,
    client: McpClient,
    config: ComponentConfig,
}

impl McpPrompt {
    /// Binds `definition` to the session it was discovered on.
    pub fn from_definition(definition: McpPromptDefinition, client: McpClient) -> Self {
        let description = definition
            .description
            .clone()
            .unwrap_or_else(|| format!("MCP prompt {}", definition.name));
        let config = ComponentConfig::new()
            .with_title(definition.name.clone())
            .with_description(description);
        Self {
            definition,
            client,
            config,
        }
    }

    /// The server's definition of this prompt.
    pub fn definition(&self) -> &McpPromptDefinition {
        &self.definition
    }

    /// Renders the prompt asynchronously.
    pub async fn get(&self, args: Map<String, Value>) -> Result<McpPromptOutput> {
        self.client.get_prompt(&self.definition.name, args).await
    }
}

impl Prompt for McpPrompt {
    type Input = McpToolInput;
    type Output = McpPromptOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn generate(&self, input: Self::Input) -> Result<Self::Output> {
        block_on_multi_thread(self.get(input.0), "McpPrompt::generate", "McpPrompt::get")
            .and_then(|r| r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{PromptMessage, Role};
    use serde_json::json;

    #[test]
    fn converts_roles_and_joins_text() {
        let result = GetPromptResult::new(vec![
            PromptMessage::new_text(Role::User, "one"),
            PromptMessage::new_text(Role::Assistant, "two"),
        ])
        .with_description("d");
        let out = McpPromptOutput::from_result(result).unwrap();
        assert_eq!(out.description.as_deref(), Some("d"));
        assert_eq!(out.content, "one\n\ntwo");
        let roles: Vec<&str> = out.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant"]);
        assert_eq!(
            out.messages[0].content,
            json!({"type": "text", "text": "one"})
        );
        assert_eq!(McpPromptOutput::schema_title(), "McpPromptOutput");
    }

    #[test]
    fn non_text_blocks_add_no_content() {
        let result = GetPromptResult::new(vec![
            PromptMessage::new(Role::User, ContentBlock::image("AAEC", "image/png")),
            PromptMessage::new_text(Role::User, "after"),
        ]);
        let out = McpPromptOutput::from_result(result).unwrap();
        assert_eq!(out.content, "after");
        assert_eq!(out.messages.len(), 2);
        assert_eq!(out.messages[0].content["type"], json!("image"));

        let only_image = GetPromptResult::new(vec![PromptMessage::new(
            Role::User,
            ContentBlock::image("AAEC", "image/png"),
        )]);
        assert_eq!(
            McpPromptOutput::from_result(only_image).unwrap().content,
            ""
        );
    }

    #[test]
    fn definition_schema_requires_only_required_arguments() {
        let mut a = rmcp::model::PromptArgument::new("a");
        a.required = Some(true);
        let mut b = rmcp::model::PromptArgument::new("b");
        b.required = Some(false);
        let c = rmcp::model::PromptArgument::new("c");
        let def = McpPromptDefinition::from(rmcp::model::Prompt::new(
            "p",
            None::<&str>,
            Some(vec![a, b, c]),
        ));
        assert_eq!(def.input_schema["required"], json!(["a"]));
        assert_eq!(def.input_schema["properties"]["c"]["type"], json!("string"));
    }
}
