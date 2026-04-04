use crate::error::Result;
use crate::schema::IOSchema;

/// Configuration for a tool, resource, or prompt component.
///
/// Allows overriding the auto-derived title and description from the schema.
#[derive(Clone, Debug, Default)]
pub struct ComponentConfig {
    /// Custom title (overrides schema title if set).
    pub title: Option<String>,
    /// Custom description (overrides schema description if set).
    pub description: Option<String>,
}

impl ComponentConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Base trait for all tools in the framework.
///
/// A tool takes structured input and produces structured output. Define input/output
/// types as associated types implementing `IOSchema`.
///
/// Mirrors Python's `BaseTool[InputSchema, OutputSchema]`.
///
/// # Example
///
/// ```ignore
/// use atomic_agents::prelude::*;
///
/// /// Calculate mathematical expressions.
/// #[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
/// pub struct CalcInput {
///     /// The expression to evaluate.
///     pub expression: String,
/// }
///
/// /// Result of a calculation.
/// #[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
/// pub struct CalcOutput {
///     /// The computed result.
///     pub result: String,
/// }
///
/// pub struct Calculator {
///     config: ComponentConfig,
/// }
///
/// impl Tool for Calculator {
///     type Input = CalcInput;
///     type Output = CalcOutput;
///
///     fn config(&self) -> &ComponentConfig { &self.config }
///
///     fn run(&self, input: Self::Input) -> Result<Self::Output> {
///         Ok(CalcOutput { result: input.expression })
///     }
/// }
/// ```
pub trait Tool: Send + Sync {
    type Input: IOSchema;
    type Output: IOSchema;

    /// Returns the component configuration.
    fn config(&self) -> &ComponentConfig;

    /// Executes the tool with the given input.
    fn run(&self, input: Self::Input) -> Result<Self::Output>;

    /// Returns the tool name (from config or input schema title).
    fn name(&self) -> &str {
        self.config()
            .title
            .as_deref()
            .unwrap_or_else(|| Self::Input::schema_title())
    }

    /// Returns the tool description (from config or input schema description).
    fn description(&self) -> &str {
        self.config()
            .description
            .as_deref()
            .unwrap_or_else(|| Self::Input::schema_description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{BasicChatInputSchema, BasicChatOutputSchema};

    struct EchoTool {
        config: ComponentConfig,
    }

    impl Tool for EchoTool {
        type Input = BasicChatInputSchema;
        type Output = BasicChatOutputSchema;

        fn config(&self) -> &ComponentConfig {
            &self.config
        }

        fn run(&self, input: Self::Input) -> Result<Self::Output> {
            Ok(BasicChatOutputSchema {
                chat_message: input.chat_message,
            })
        }
    }

    #[test]
    fn test_tool_run() {
        let tool = EchoTool {
            config: ComponentConfig::default(),
        };
        let result = tool
            .run(BasicChatInputSchema {
                chat_message: "echo this".to_string(),
            })
            .unwrap();
        assert_eq!(result.chat_message, "echo this");
    }

    #[test]
    fn test_tool_name_from_schema() {
        let tool = EchoTool {
            config: ComponentConfig::default(),
        };
        assert_eq!(tool.name(), "BasicChatInputSchema");
    }

    #[test]
    fn test_tool_name_from_config() {
        let tool = EchoTool {
            config: ComponentConfig::new().with_title("Echo"),
        };
        assert_eq!(tool.name(), "Echo");
    }

    #[test]
    fn test_tool_description_from_config() {
        let tool = EchoTool {
            config: ComponentConfig::new().with_description("Echoes input"),
        };
        assert_eq!(tool.description(), "Echoes input");
    }
}
