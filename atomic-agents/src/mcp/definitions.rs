//! Tool, resource and prompt metadata discovered from an MCP server.

use rmcp::model::{Prompt, Resource, ResourceTemplate, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// A tool advertised by an MCP server (`tools/list`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// Tool name, used in `tools/call`.
    pub name: String,
    /// Human-readable display title.
    pub title: Option<String>,
    /// What the tool does.
    pub description: Option<String>,
    /// JSON Schema of the arguments, exactly as the server sent it.
    pub input_schema: Value,
    /// JSON Schema of the structured result, if the server declares one.
    pub output_schema: Option<Value>,
}

impl From<Tool> for McpToolDefinition {
    fn from(tool: Tool) -> Self {
        Self {
            name: tool.name.into_owned(),
            title: tool.title,
            description: tool.description.map(|d| d.into_owned()),
            input_schema: Value::Object((*tool.input_schema).clone()),
            output_schema: tool.output_schema.map(|s| Value::Object((*s).clone())),
        }
    }
}

/// A resource or resource template advertised by an MCP server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpResourceDefinition {
    /// Resource name.
    pub name: String,
    /// What the resource contains.
    pub description: Option<String>,
    /// Concrete URI, or an RFC 6570 URI template when `is_template` is set.
    pub uri: String,
    /// MIME type of the contents, if known.
    pub mime_type: Option<String>,
    /// Whether `uri` is a template whose placeholders must be filled.
    pub is_template: bool,
    /// JSON Schema of the template parameters (an empty object schema for
    /// concrete resources).
    pub input_schema: Value,
}

impl McpResourceDefinition {
    /// A concrete resource from `resources/list`.
    pub fn from_resource(resource: Resource) -> Self {
        Self {
            name: resource.name,
            description: resource.description,
            uri: resource.uri,
            mime_type: resource.mime_type,
            is_template: false,
            input_schema: object_schema(Map::new(), Vec::new()),
        }
    }

    /// A resource template from `resources/templates/list`.
    ///
    /// Every template variable becomes a string property. Simple, `{+..}`,
    /// `{#..}` and path-style (`{.x}`, `{/..}`, `{;..}`) variables are
    /// required; form-style query variables (`{?..}`, `{&..}`) are optional.
    pub fn from_template(template: ResourceTemplate) -> Self {
        let mut properties = Map::new();
        let mut required = Vec::new();
        for (name, is_required) in template_variables(&template.uri_template) {
            properties.insert(
                name.clone(),
                json!({"type": "string", "description": format!("URI parameter {name}")}),
            );
            if is_required {
                required.push(Value::String(name));
            }
        }
        Self {
            name: template.name,
            description: template.description,
            uri: template.uri_template,
            mime_type: template.mime_type,
            is_template: true,
            input_schema: object_schema(properties, required),
        }
    }
}

/// An argument accepted by an MCP prompt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpPromptArgument {
    /// Argument name.
    pub name: String,
    /// What the argument means.
    pub description: Option<String>,
    /// Whether the prompt requires it.
    pub required: bool,
}

/// A prompt template advertised by an MCP server (`prompts/list`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpPromptDefinition {
    /// Prompt name, used in `prompts/get`.
    pub name: String,
    /// What the prompt is for.
    pub description: Option<String>,
    /// Declared arguments, in server order.
    pub arguments: Vec<McpPromptArgument>,
    /// JSON Schema of the arguments (all strings).
    pub input_schema: Value,
}

impl From<Prompt> for McpPromptDefinition {
    fn from(prompt: Prompt) -> Self {
        let arguments: Vec<McpPromptArgument> = prompt
            .arguments
            .unwrap_or_default()
            .into_iter()
            .map(|a| McpPromptArgument {
                name: a.name,
                description: a.description,
                required: a.required == Some(true),
            })
            .collect();
        let mut properties = Map::new();
        let mut required = Vec::new();
        for arg in &arguments {
            let mut prop = json!({"type": "string"});
            if let Some(d) = &arg.description {
                prop["description"] = Value::String(d.clone());
            }
            properties.insert(arg.name.clone(), prop);
            if arg.required {
                required.push(Value::String(arg.name.clone()));
            }
        }
        Self {
            name: prompt.name,
            description: prompt.description,
            arguments,
            input_schema: object_schema(properties, required),
        }
    }
}

fn object_schema(properties: Map<String, Value>, required: Vec<Value>) -> Value {
    json!({"type": "object", "properties": properties, "required": required})
}

/// Variable names in an RFC 6570 template, deduplicated in order of first
/// appearance, each with whether it is required.
fn template_variables(template: &str) -> Vec<(String, bool)> {
    let mut vars: Vec<(String, bool)> = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else { break };
        let expr = &after[..close];
        rest = &after[close + 1..];

        let (operator, list) = match expr.chars().next() {
            Some(c @ ('+' | '#' | '.' | '/' | ';' | '?' | '&')) => (Some(c), &expr[1..]),
            _ => (None, expr),
        };
        let required = !matches!(operator, Some('?' | '&'));
        for spec in list.split(',') {
            // Drop value modifiers: prefix (`var:3`) and explode (`var*`).
            let name = spec
                .split(':')
                .next()
                .unwrap_or("")
                .trim_end_matches('*')
                .trim();
            if name.is_empty() {
                continue;
            }
            match vars.iter_mut().find(|(n, _)| n == name) {
                Some((_, req)) => *req |= required,
                None => vars.push((name.to_string(), required)),
            }
        }
    }
    vars
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn keys(schema: &Value) -> Vec<String> {
        let mut k: Vec<String> = schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        k.sort();
        k
    }

    #[test]
    fn template_placeholders_become_properties() {
        let def = McpResourceDefinition::from_template(ResourceTemplate::new(
            "mem://x/{a}/{+b}{?c,d}",
            "x",
        ));
        assert!(def.is_template);
        assert_eq!(def.uri, "mem://x/{a}/{+b}{?c,d}");
        assert_eq!(keys(&def.input_schema), ["a", "b", "c", "d"]);
        assert_eq!(def.input_schema["required"], json!(["a", "b"]));
        assert_eq!(
            def.input_schema["properties"]["a"],
            json!({"type": "string", "description": "URI parameter a"})
        );
    }

    #[test]
    fn template_fragment_required_and_continuation_optional() {
        let vars = template_variables("x{#f}{?q}{&r}{id}{id}");
        assert_eq!(
            vars,
            [
                ("f".to_string(), true),
                ("q".to_string(), false),
                ("r".to_string(), false),
                ("id".to_string(), true),
            ]
        );
    }

    #[test]
    fn template_modifiers_are_stripped() {
        let vars = template_variables("{/path*}{var:3}");
        assert_eq!(
            vars,
            [("path".to_string(), true), ("var".to_string(), true)]
        );
    }

    #[test]
    fn concrete_resource_has_no_properties() {
        let def = McpResourceDefinition::from_resource(
            Resource::new("mem://x/%7Ba%7D", "x").with_mime_type("text/plain"),
        );
        assert!(!def.is_template);
        assert_eq!(def.uri, "mem://x/%7Ba%7D");
        assert_eq!(def.mime_type.as_deref(), Some("text/plain"));
        assert_eq!(
            def.input_schema,
            json!({"type": "object", "properties": {}, "required": []})
        );
    }

    #[test]
    fn tool_schemas_pass_through_unchanged() {
        let input = json!({
            "type": "object",
            "$defs": {"P": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "properties": {"p": {"$ref": "#/$defs/P"}},
            "additionalProperties": false,
            "x-vendor": {"anything": [1, 2]}
        });
        let output = json!({"type": "object", "properties": {"n": {"type": "number"}}});
        let tool = Tool::new("t", "desc", input.as_object().unwrap().clone())
            .with_raw_output_schema(Arc::new(output.as_object().unwrap().clone()))
            .with_title("Title");
        let def = McpToolDefinition::from(tool);
        assert_eq!(def.name, "t");
        assert_eq!(def.title.as_deref(), Some("Title"));
        assert_eq!(def.description.as_deref(), Some("desc"));
        assert_eq!(def.input_schema, input);
        assert_eq!(def.output_schema, Some(output));
    }

    #[test]
    fn prompt_arguments_become_schema() {
        let mut name = rmcp::model::PromptArgument::new("name").with_description("Who");
        name.required = Some(true);
        let tone = rmcp::model::PromptArgument::new("tone");
        let def = McpPromptDefinition::from(Prompt::new(
            "greet",
            Some("Greets someone"),
            Some(vec![name, tone]),
        ));
        assert_eq!(def.description.as_deref(), Some("Greets someone"));
        assert_eq!(def.arguments.len(), 2);
        assert!(def.arguments[0].required && !def.arguments[1].required);
        assert_eq!(keys(&def.input_schema), ["name", "tone"]);
        assert_eq!(def.input_schema["required"], json!(["name"]));
        assert_eq!(
            def.input_schema["properties"]["name"],
            json!({"type": "string", "description": "Who"})
        );
    }
}
