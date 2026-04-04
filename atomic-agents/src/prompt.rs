use crate::error::Result;
use crate::schema::IOSchema;
use crate::tool::ComponentConfig;

/// Base trait for prompt templates in the framework.
///
/// A prompt generates structured output from structured input parameters.
/// Mirrors Python's `BasePrompt[InputSchema, OutputSchema]`.
pub trait Prompt: Send + Sync {
    type Input: IOSchema;
    type Output: IOSchema;

    /// Returns the component configuration.
    fn config(&self) -> &ComponentConfig;

    /// Generates output from the given input parameters.
    fn generate(&self, input: Self::Input) -> Result<Self::Output>;

    /// Returns the prompt name (from config or input schema title).
    fn name(&self) -> &str {
        self.config()
            .title
            .as_deref()
            .unwrap_or_else(|| Self::Input::schema_title())
    }

    /// Returns the prompt description (from config or input schema description).
    fn description(&self) -> &str {
        self.config()
            .description
            .as_deref()
            .unwrap_or_else(|| Self::Input::schema_description())
    }
}
