use crate::error::Result;
use crate::schema::IOSchema;
use crate::tool::ComponentConfig;

/// Base trait for resource readers in the framework.
///
/// A resource reads data from an external source and returns structured output.
/// Mirrors Python's `BaseResource[InputSchema, OutputSchema]`.
pub trait Resource: Send + Sync {
    type Input: IOSchema;
    type Output: IOSchema;

    /// Returns the component configuration.
    fn config(&self) -> &ComponentConfig;

    /// Reads the resource with the given input parameters.
    fn read(&self, input: Self::Input) -> Result<Self::Output>;

    /// Returns the resource name (from config or input schema title).
    fn name(&self) -> &str {
        self.config()
            .title
            .as_deref()
            .unwrap_or_else(|| Self::Input::schema_title())
    }

    /// Returns the resource description (from config or input schema description).
    fn description(&self) -> &str {
        self.config()
            .description
            .as_deref()
            .unwrap_or_else(|| Self::Input::schema_description())
    }
}
