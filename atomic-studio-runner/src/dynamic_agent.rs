use std::collections::HashMap;

use atomic_agents::client::LlmClient;
use atomic_agents::client::types::{
    ChatMessage, ChatRequest, FunctionDefinition, ResponseFormat, ToolDefinition,
};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::pipeline::{AgentNodeConfig, SystemPromptConfig};

/// A dynamically-configured agent that uses `serde_json::Value` for I/O
/// instead of compile-time `IOSchema` types.
///
/// This mirrors `AtomicAgent::run_async` but works with JSON Schema definitions
/// loaded at runtime, following the same pattern as `McpTool`.
pub struct DynamicAgent {
    client: Box<dyn LlmClient>,
    model: String,
    input_schema: Value,
    output_schema: Value,
    output_title: String,
    output_description: String,
    system_prompt: String,
    extraction_mode: ExtractionMode,
    max_retries: usize,
    extra_params: HashMap<String, Value>,
    assistant_role: String,
    system_role: String,
    messages: Vec<ChatMessage>,
}

#[derive(Clone, Debug)]
enum ExtractionMode {
    Tools,
    Json,
    MdJson,
}

impl DynamicAgent {
    /// Creates a new DynamicAgent from an agent node config and resolved schemas.
    pub fn new(
        client: Box<dyn LlmClient>,
        config: &AgentNodeConfig,
        model: String,
        input_schema: Value,
        output_schema: Value,
    ) -> Self {
        let output_title = output_schema
            .get("title")
            .and_then(|t| t.as_str())
            .unwrap_or("Output")
            .to_string();
        let output_description = output_schema
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("Structured output")
            .to_string();

        let system_prompt = build_system_prompt(&config.system_prompt);
        let extraction_mode = match config.extraction_mode.as_str() {
            "json" => ExtractionMode::Json,
            "md_json" => ExtractionMode::MdJson,
            _ => ExtractionMode::Tools,
        };

        let assistant_role = client.assistant_role().to_string();
        let system_role = "system".to_string();

        Self {
            client,
            model,
            input_schema,
            output_schema,
            output_title,
            output_description,
            system_prompt,
            extraction_mode,
            max_retries: config.max_retries,
            extra_params: config.extra_params.clone(),
            assistant_role,
            system_role,
            messages: Vec::new(),
        }
    }

    /// Executes the agent with the given input value.
    ///
    /// Validates input against the input schema, sends to the LLM with
    /// output schema constraints, validates the response, and retries
    /// on parse failures.
    pub async fn run(&mut self, input: Value) -> Result<Value> {
        // Validate input against schema
        validate_against_schema(&input, &self.input_schema)?;

        // Add user message
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: Value::String(serde_json::to_string(&input).unwrap_or_default()),
        });

        let mut attempts = 0;
        loop {
            let request = self.build_request();

            let response = self
                .client
                .chat_completion(request)
                .await
                .map_err(Error::Agent)?;

            // Try to parse the response as JSON
            match serde_json::from_str::<Value>(&response.content) {
                Ok(output) if validate_against_schema(&output, &self.output_schema).is_ok() => {
                    self.messages.push(ChatMessage {
                        role: self.assistant_role.clone(),
                        content: Value::String(response.content),
                    });
                    return Ok(output);
                }
                Ok(_invalid) => {
                    if attempts >= self.max_retries {
                        return Err(Error::NodeExecution {
                            node_id: String::new(),
                            message: format!(
                                "Output schema validation failed after {} attempts. Last response: {}",
                                attempts,
                                &response.content.chars().take(500).collect::<String>()
                            ),
                        });
                    }
                    attempts += 1;
                    self.append_retry_messages(
                        &response.content,
                        "Output did not match the required JSON schema.",
                    );
                }
                Err(e) => {
                    if attempts >= self.max_retries {
                        return Err(Error::NodeExecution {
                            node_id: String::new(),
                            message: format!(
                                "Failed to parse response as JSON after {} attempts: {}. Last response: {}",
                                attempts,
                                e,
                                &response.content.chars().take(500).collect::<String>()
                            ),
                        });
                    }
                    attempts += 1;
                    self.append_retry_messages(&response.content, &format!("Invalid JSON: {e}"));
                }
            }
        }
    }

    fn build_request(&self) -> ChatRequest {
        let mut messages = Vec::new();

        // System message
        let mut prompt = self.system_prompt.clone();

        if matches!(
            self.extraction_mode,
            ExtractionMode::Json | ExtractionMode::MdJson
        ) {
            let schema_json = serde_json::to_string_pretty(&self.output_schema).unwrap_or_default();
            prompt.push_str(&format!(
                "\n\n# RESPONSE SCHEMA\nYou must respond with JSON matching this schema:\n```json\n{}\n```\n",
                schema_json
            ));
        }

        messages.push(ChatMessage {
            role: self.system_role.clone(),
            content: Value::String(prompt),
        });

        // Conversation messages
        messages.extend(self.messages.clone());

        let tools = match self.extraction_mode {
            ExtractionMode::Tools => {
                let mut params = self.output_schema.clone();
                if let Value::Object(ref mut map) = params {
                    map.remove("$schema");
                }
                Some(vec![ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: self.output_title.clone(),
                        description: Some(self.output_description.clone()),
                        parameters: params,
                        strict: None,
                    },
                }])
            }
            _ => None,
        };

        let response_format = match self.extraction_mode {
            ExtractionMode::Json => Some(ResponseFormat {
                format_type: "json_object".to_string(),
                json_schema: None,
            }),
            _ => None,
        };

        let mut request = ChatRequest::new(self.model.clone(), messages);
        request.tools = tools;
        request.response_format = response_format;
        request.extra = self.extra_params.clone();
        request
    }

    fn append_retry_messages(&mut self, response_content: &str, error_msg: &str) {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: Value::String(response_content.to_string()),
        });
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: Value::String(format!(
                "Your response did not match the required JSON schema. Error: {}. \
                 Please try again and respond with valid JSON matching the schema.",
                error_msg
            )),
        });
    }
}

/// Validates a JSON value against a JSON Schema.
pub fn validate_against_schema(value: &Value, schema: &Value) -> Result<()> {
    let validator = jsonschema::validator_for(schema)
        .map_err(|e| Error::SchemaValidation(format!("Invalid schema: {e}")))?;

    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|e| e.to_string())
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(Error::SchemaValidation(errors.join("; ")))
    }
}

fn build_system_prompt(config: &SystemPromptConfig) -> String {
    let mut sections = Vec::new();

    if !config.background.is_empty() {
        let mut section = String::from("# IDENTITY and PURPOSE\n");
        for item in &config.background {
            section.push_str(&format!("- {item}\n"));
        }
        sections.push(section);
    }

    if !config.steps.is_empty() {
        let mut section = String::from("# INTERNAL ASSISTANT STEPS\n");
        for item in &config.steps {
            section.push_str(&format!("- {item}\n"));
        }
        sections.push(section);
    }

    if !config.output_instructions.is_empty() {
        let mut section = String::from("# OUTPUT INSTRUCTIONS\n");
        for item in &config.output_instructions {
            section.push_str(&format!("- {item}\n"));
        }
        sections.push(section);
    }

    if sections.is_empty() {
        "You are a helpful AI assistant. Always respond using the proper JSON schema.".to_string()
    } else {
        sections.push(
            "# OUTPUT INSTRUCTIONS\n- Always respond using the proper JSON schema.\n".to_string(),
        );
        sections.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_against_schema_valid() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });
        let value = json!({"name": "Alice", "age": 30});
        assert!(validate_against_schema(&value, &schema).is_ok());
    }

    #[test]
    fn test_validate_against_schema_invalid() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        let value = json!({"age": 30});
        assert!(validate_against_schema(&value, &schema).is_err());
    }

    #[test]
    fn test_build_system_prompt() {
        let config = SystemPromptConfig {
            background: vec!["You classify tickets.".to_string()],
            steps: vec!["Read the ticket.".to_string(), "Classify it.".to_string()],
            output_instructions: vec![],
        };
        let prompt = build_system_prompt(&config);
        assert!(prompt.contains("IDENTITY and PURPOSE"));
        assert!(prompt.contains("classify tickets"));
        assert!(prompt.contains("INTERNAL ASSISTANT STEPS"));
    }

    #[test]
    fn test_build_system_prompt_empty() {
        let config = SystemPromptConfig::default();
        let prompt = build_system_prompt(&config);
        assert!(prompt.contains("helpful AI assistant"));
    }
}
