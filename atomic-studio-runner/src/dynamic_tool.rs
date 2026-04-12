use async_trait::async_trait;
use serde_json::Value;

use atomic_agents::schema::IOSchema;
use atomic_agents::tool::Tool;

use crate::error::{Error, Result};

/// A tool that operates on `serde_json::Value` I/O, enabling runtime-discovered
/// tools to be used in pipelines without compile-time type information.
#[async_trait]
pub trait DynamicToolRunner: Send + Sync {
    /// Tool name.
    fn name(&self) -> &str;

    /// JSON Schema for the tool's input.
    fn input_schema(&self) -> &Value;

    /// JSON Schema for the tool's output.
    fn output_schema(&self) -> Option<&Value>;

    /// Executes the tool with the given input.
    async fn run(&self, input: Value) -> Result<Value>;
}

/// Wraps a typed `Tool` implementation to expose it as a `DynamicToolRunner`.
///
/// Deserializes `Value` input into the tool's `Input` type, calls `run()`,
/// and serializes the `Output` back to `Value`.
pub struct ToolAdapter<T: Tool> {
    tool: T,
    input_schema: Value,
    output_schema: Value,
}

impl<T: Tool> ToolAdapter<T> {
    pub fn new(tool: T) -> Self {
        let input_schema =
            serde_json::to_value(T::Input::generate_schema()).unwrap_or_default();
        let output_schema =
            serde_json::to_value(T::Output::generate_schema()).unwrap_or_default();
        Self {
            tool,
            input_schema,
            output_schema,
        }
    }
}

#[async_trait]
impl<T: Tool + 'static> DynamicToolRunner for ToolAdapter<T> {
    fn name(&self) -> &str {
        self.tool.name()
    }

    fn input_schema(&self) -> &Value {
        &self.input_schema
    }

    fn output_schema(&self) -> Option<&Value> {
        Some(&self.output_schema)
    }

    async fn run(&self, input: Value) -> Result<Value> {
        let typed_input: T::Input = serde_json::from_value(input)
            .map_err(|e| Error::NodeExecution {
                node_id: String::new(),
                message: format!("Failed to deserialize tool input: {e}"),
            })?;

        let output = self.tool.run(typed_input).map_err(Error::Agent)?;

        serde_json::to_value(output)
            .map_err(|e| Error::NodeExecution {
                node_id: String::new(),
                message: format!("Failed to serialize tool output: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomic_agents::schema::{BasicChatInputSchema, BasicChatOutputSchema};
    use atomic_agents::tool::ComponentConfig;
    use serde_json::json;

    struct EchoTool {
        config: ComponentConfig,
    }

    impl Tool for EchoTool {
        type Input = BasicChatInputSchema;
        type Output = BasicChatOutputSchema;

        fn config(&self) -> &ComponentConfig {
            &self.config
        }

        fn run(&self, input: Self::Input) -> atomic_agents::error::Result<Self::Output> {
            Ok(BasicChatOutputSchema {
                chat_message: format!("Echo: {}", input.chat_message),
            })
        }
    }

    #[tokio::test]
    async fn test_tool_adapter() {
        let tool = EchoTool {
            config: ComponentConfig::new().with_title("Echo"),
        };
        let adapter = ToolAdapter::new(tool);

        assert_eq!(adapter.name(), "Echo");

        let input = json!({"chat_message": "hello"});
        let output = adapter.run(input).await.unwrap();
        assert_eq!(output["chat_message"], "Echo: hello");
    }

    #[tokio::test]
    async fn test_tool_adapter_invalid_input() {
        let tool = EchoTool {
            config: ComponentConfig::default(),
        };
        let adapter = ToolAdapter::new(tool);

        let input = json!({"wrong_field": "hello"});
        let result = adapter.run(input).await;
        assert!(result.is_err());
    }
}
