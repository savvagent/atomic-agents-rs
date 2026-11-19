use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

/// Marker trait for all input/output schemas in the framework.
///
/// This trait combines JSON serialization (serde), JSON Schema generation (schemars),
/// and schema metadata. Implement via `#[derive(IOSchema)]` which enforces that a doc
/// comment is present (used as the schema description).
///
/// # Example
///
/// ```
/// use atomic_agents::prelude::*;
/// use serde::{Serialize, Deserialize};
/// use schemars::JsonSchema;
///
/// /// A simple chat message.
/// #[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
/// pub struct ChatMessage {
///     /// The message content.
///     pub text: String,
/// }
/// ```
pub trait IOSchema:
    Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + std::fmt::Debug + 'static
{
    /// Returns the schema title (struct name).
    fn schema_title() -> &'static str;

    /// Returns the schema description (from doc comment).
    fn schema_description() -> &'static str;

    /// Generates the full JSON Schema for this type.
    fn generate_schema() -> schemars::schema::RootSchema {
        let settings = schemars::r#gen::SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
        });
        let generator = settings.into_generator();
        generator.into_root_schema_for::<Self>()
    }

    /// Serializes this value to a JSON string.
    fn to_json(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserializes a value from a JSON string.
    fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serializes this value to a `serde_json::Value`.
    fn to_value(&self) -> std::result::Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(self)
    }
}

/// Default input schema for basic chat agents.
///
/// Mirrors Python's `BasicChatInputSchema`.
#[derive(Clone, Debug, Serialize, serde::Deserialize, JsonSchema)]
pub struct BasicChatInputSchema {
    /// The chat message from the user.
    pub chat_message: String,
}

impl IOSchema for BasicChatInputSchema {
    fn schema_title() -> &'static str {
        "BasicChatInputSchema"
    }

    fn schema_description() -> &'static str {
        "Default input schema for basic chat agents."
    }
}

/// Default output schema for basic chat agents.
///
/// Mirrors Python's `BasicChatOutputSchema`.
#[derive(Clone, Debug, Serialize, serde::Deserialize, JsonSchema)]
pub struct BasicChatOutputSchema {
    /// The chat message response. Supports markdown formatting.
    pub chat_message: String,
}

impl IOSchema for BasicChatOutputSchema {
    fn schema_title() -> &'static str {
        "BasicChatOutputSchema"
    }

    fn schema_description() -> &'static str {
        "Default output schema for basic chat agents."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_input_schema_json_roundtrip() {
        let input = BasicChatInputSchema {
            chat_message: "Hello, world!".to_string(),
        };
        let json = input.to_json().unwrap();
        let deserialized = BasicChatInputSchema::from_json(&json).unwrap();
        assert_eq!(input.chat_message, deserialized.chat_message);
    }

    #[test]
    fn test_basic_output_schema_json_roundtrip() {
        let output = BasicChatOutputSchema {
            chat_message: "Hi there!".to_string(),
        };
        let json = output.to_json().unwrap();
        let deserialized = BasicChatOutputSchema::from_json(&json).unwrap();
        assert_eq!(output.chat_message, deserialized.chat_message);
    }

    #[test]
    fn test_json_schema_generation() {
        let schema = BasicChatInputSchema::generate_schema();
        let schema_json = serde_json::to_value(&schema).unwrap();
        assert!(schema_json.get("properties").is_some());
        assert!(schema_json["properties"].get("chat_message").is_some());
    }

    #[test]
    fn test_schema_metadata() {
        assert_eq!(BasicChatInputSchema::schema_title(), "BasicChatInputSchema");
        assert!(!BasicChatInputSchema::schema_description().is_empty());
    }
}
