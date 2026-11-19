//! [`McpTool`]: a discovered MCP tool exposed through the framework's [`Tool`] trait.

use std::future::Future;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::runtime::{Handle, RuntimeFlavor};

use super::client::McpClient;
use super::definitions::McpToolDefinition;
use crate::error::{Error, Result};
use crate::schema::IOSchema;
use crate::tool::{ComponentConfig, Tool};

/// Arguments for an MCP tool call, resource template or prompt, as a JSON
/// object. It is the input of [`McpTool`], [`McpResource`](super::McpResource)
/// (template variables) and [`McpPrompt`](super::McpPrompt) (prompt
/// arguments).
///
/// Serializes transparently as the object itself.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct McpToolInput(pub Map<String, Value>);

impl From<Map<String, Value>> for McpToolInput {
    fn from(args: Map<String, Value>) -> Self {
        Self(args)
    }
}

impl IOSchema for McpToolInput {
    fn schema_title() -> &'static str {
        "McpToolInput"
    }
    fn schema_description() -> &'static str {
        "Arguments for an MCP tool call, resource template or prompt."
    }
}

/// Result of an MCP tool call, converted to JSON (see [`McpClient::call_tool`]).
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct McpToolOutput {
    /// The converted tool result.
    pub result: Value,
}

impl IOSchema for McpToolOutput {
    fn schema_title() -> &'static str {
        "McpToolOutput"
    }
    fn schema_description() -> &'static str {
        "Result of an MCP tool call."
    }
}

/// An MCP server tool bound to the session it was discovered on.
///
/// Implements [`Tool`] with dynamic JSON input and output. [`Tool::run`] is
/// synchronous, so it blocks the calling thread and requires a multi-threaded
/// tokio runtime; in async code use [`McpTool::call`].
#[derive(Clone, Debug)]
pub struct McpTool {
    definition: McpToolDefinition,
    client: McpClient,
    config: ComponentConfig,
}

impl McpTool {
    /// Binds `definition` to the session it was discovered on.
    pub fn from_definition(definition: McpToolDefinition, client: McpClient) -> Self {
        let description = definition
            .description
            .clone()
            .unwrap_or_else(|| format!("MCP tool {}", definition.name));
        let config = ComponentConfig::new()
            .with_title(definition.name.clone())
            .with_description(description);
        Self {
            definition,
            client,
            config,
        }
    }

    /// The server's definition of this tool.
    pub fn definition(&self) -> &McpToolDefinition {
        &self.definition
    }

    /// The tool name used in `tools/call`.
    pub fn mcp_name(&self) -> &str {
        &self.definition.name
    }

    /// JSON Schema of the arguments, as the server sent it.
    pub fn input_schema(&self) -> &Value {
        &self.definition.input_schema
    }

    /// JSON Schema of the structured result, if the server declares one.
    pub fn output_schema(&self) -> Option<&Value> {
        self.definition.output_schema.as_ref()
    }

    /// Calls the tool asynchronously.
    pub async fn call(&self, args: Map<String, Value>) -> Result<Value> {
        self.client.call_tool(&self.definition.name, args).await
    }
}

impl Tool for McpTool {
    type Input = McpToolInput;
    type Output = McpToolOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        block_on_multi_thread(self.call(input.0), "McpTool::run", "McpTool::call")
            .and_then(|r| r.map(|result| McpToolOutput { result }))
    }
}

/// Drives `fut` to completion from synchronous code without panicking.
///
/// `block_in_place` panics on a current-thread runtime and `Handle::current`
/// panics outside any runtime, so both cases are reported as an error instead.
/// `sync_name` and `async_name` name the blocking entry point and its async
/// counterpart in that error.
pub(crate) fn block_on_multi_thread<F: Future>(
    fut: F,
    sync_name: &str,
    async_name: &str,
) -> Result<F::Output> {
    match Handle::try_current() {
        Ok(h) if h.runtime_flavor() == RuntimeFlavor::MultiThread => {
            Ok(tokio::task::block_in_place(|| h.block_on(fut)))
        }
        _ => Err(Error::Mcp(format!(
            "{sync_name} requires a multi-threaded tokio runtime; use {async_name} in async code"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn input_is_transparent() {
        let input: McpToolInput = serde_json::from_value(json!({"a": 1})).unwrap();
        assert_eq!(input.0["a"], json!(1));
        assert_eq!(serde_json::to_value(&input).unwrap(), json!({"a": 1}));
        assert!(McpToolInput::default().0.is_empty());
    }

    #[test]
    fn schema_metadata() {
        assert_eq!(McpToolInput::schema_title(), "McpToolInput");
        assert_eq!(McpToolOutput::schema_title(), "McpToolOutput");
        assert_eq!(
            McpToolInput::schema_description(),
            "Arguments for an MCP tool call, resource template or prompt."
        );
        assert_eq!(
            McpToolOutput::schema_description(),
            "Result of an MCP tool call."
        );
        let schema = serde_json::to_value(McpToolInput::generate_schema()).unwrap();
        assert_eq!(schema["type"], json!("object"));
    }

    #[test]
    fn blocking_outside_a_runtime_errors() {
        let err = block_on_multi_thread(async { 1 }, "A::run", "A::call").unwrap_err();
        assert!(err.to_string().contains("multi-threaded"), "{err}");
        assert_eq!(
            err.to_string().trim_start_matches("MCP error: "),
            "A::run requires a multi-threaded tokio runtime; use A::call in async code"
        );
    }

    #[tokio::test]
    async fn blocking_on_current_thread_errors() {
        let err = block_on_multi_thread(async { 1 }, "A::run", "A::call").unwrap_err();
        assert!(err.to_string().contains("multi-threaded"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn blocking_on_multi_thread_runs() {
        assert_eq!(
            block_on_multi_thread(async { 7 }, "A::run", "A::call").unwrap(),
            7
        );
    }
}
