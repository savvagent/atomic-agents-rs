//! Orchestrator schema over a set of MCP tools, and dispatch of the model's choice.
//!
//! [`orchestrator_schema`] builds one JSON Schema whose `tool_parameters`
//! is an `anyOf` of every tool's argument schema, each pinned to its tool by
//! a leading `tool_name` enum. Hand it to
//! [`AgentConfig::with_output_schema`](crate::agent::AgentConfig::with_output_schema)
//! with [`McpOrchestratorOutput`] as the agent's output type; the LLM then sees
//! the real per-tool schemas, and [`McpToolset::dispatch`] runs the call it
//! picked.
//!
//! The schema is deterministic: `serde_json` runs with `preserve_order` under
//! the `mcp` feature, so `tool_name` stays the first property and tools keep
//! their discovery order.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::client::McpClient;
use super::prompt::McpPrompt;
use super::resource::McpResource;
use super::tool::McpTool;
use super::transport::McpTransportType;
use crate::error::{Error, Result};
use crate::schema::IOSchema;

/// A tool selected by the model: its name plus the tool's own arguments.
///
/// `arguments` is flattened, so the wire shape is
/// `{"tool_name": "add", "a": 1, "b": 2}`.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct McpToolCall {
    /// Name of the MCP tool to call.
    pub tool_name: String,
    /// The tool's arguments (every property other than `tool_name`).
    #[serde(flatten)]
    pub arguments: Map<String, Value>,
}

impl IOSchema for McpToolCall {
    fn schema_title() -> &'static str {
        "McpToolCall"
    }
    fn schema_description() -> &'static str {
        "A selected MCP tool and its arguments."
    }
}

/// Output of an MCP orchestrator agent.
///
/// The derived schema is only a fallback: agents should be configured with
/// [`orchestrator_schema`] through `AgentConfig::with_output_schema`, and that
/// override is what the LLM sees.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct McpOrchestratorOutput {
    /// The parameters for the selected tool.
    pub tool_parameters: McpToolCall,
}

impl IOSchema for McpOrchestratorOutput {
    fn schema_title() -> &'static str {
        "McpOrchestratorOutput"
    }
    fn schema_description() -> &'static str {
        "Output schema for the MCP orchestrator. Contains the parameters for the selected tool."
    }
}

/// Builds the orchestrator output schema for `tools`.
///
/// See the module docs. Errors when `tools` is empty, two tools share a name,
/// or a tool already declares a `tool_name` property.
pub fn orchestrator_schema(tools: &[McpTool]) -> Result<Value> {
    let entries: Vec<(&str, Option<&str>, &Value)> = tools
        .iter()
        .map(|t| {
            (
                t.mcp_name(),
                t.definition().description.as_deref(),
                t.input_schema(),
            )
        })
        .collect();
    build_orchestrator_schema(&entries)
}

/// Alias of [`orchestrator_schema`], named after the upstream Python helper.
pub fn create_mcp_orchestrator_schema(tools: &[McpTool]) -> Result<Value> {
    orchestrator_schema(tools)
}

/// Replaces every character outside `[A-Za-z0-9_]` with `_`.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Rewrites `#/$defs/X...` and `#/definitions/X...` refs to `#/$defs/{prefix}__X...`.
fn rewrite_refs(value: &mut Value, prefix: &str) {
    match value {
        Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                match (key.as_str(), &*v) {
                    ("$ref", Value::String(r)) => {
                        let rest = r
                            .strip_prefix("#/$defs/")
                            .or_else(|| r.strip_prefix("#/definitions/"));
                        if let Some(rest) = rest {
                            *v = Value::String(format!("#/$defs/{prefix}__{rest}"));
                        }
                    }
                    _ => rewrite_refs(v, prefix),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|v| rewrite_refs(v, prefix)),
        _ => {}
    }
}

/// Pure schema construction over `(name, description, input_schema)` triples.
pub(crate) fn build_orchestrator_schema(entries: &[(&str, Option<&str>, &Value)]) -> Result<Value> {
    if entries.is_empty() {
        return Err(Error::Mcp(
            "orchestrator schema needs at least one tool".to_string(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    let mut variants = Vec::with_capacity(entries.len());
    let mut root_defs = Map::new();

    for &(name, description, input_schema) in entries {
        if !seen.insert(name) {
            return Err(Error::Mcp(format!("duplicate MCP tool name {name:?}")));
        }
        let mut schema = match input_schema {
            Value::Object(map) => map.clone(),
            _ => Map::new(),
        };
        let mut properties = match schema.remove("properties") {
            Some(Value::Object(p)) => p,
            _ => Map::new(),
        };
        if properties.contains_key("tool_name") {
            return Err(Error::Mcp(format!(
                "MCP tool {name:?} declares a `tool_name` property, which the orchestrator reserves"
            )));
        }

        let prefix = sanitize(name);
        let mut defs = Map::new();
        for key in ["$defs", "definitions"] {
            if let Some(Value::Object(found)) = schema.remove(key) {
                defs.extend(found);
            }
        }
        let mut body = Value::Object(schema);
        let mut props_value = Value::Object(properties);
        rewrite_refs(&mut body, &prefix);
        rewrite_refs(&mut props_value, &prefix);
        for (def_name, mut def) in defs {
            rewrite_refs(&mut def, &prefix);
            root_defs.insert(format!("{prefix}__{def_name}"), def);
        }
        let Value::Object(mut schema) = body else {
            unreachable!("body was built from a map")
        };
        let Value::Object(old_properties) = props_value else {
            unreachable!("properties was built from a map")
        };
        properties = Map::new();
        properties.insert(
            "tool_name".to_string(),
            json!({
                "type": "string",
                "enum": [name],
                "description": format!("Required identifier for the {name} tool."),
            }),
        );
        properties.extend(old_properties);

        let mut required = vec![Value::String("tool_name".to_string())];
        if let Some(Value::Array(existing)) = schema.remove("required") {
            for r in existing {
                if !required.contains(&r) {
                    required.push(r);
                }
            }
        }

        schema.insert("type".to_string(), json!("object"));
        schema.insert("title".to_string(), json!(name));
        if let Some(d) = description
            && !schema.contains_key("description")
        {
            schema.insert("description".to_string(), json!(d));
        }
        schema.insert("properties".to_string(), Value::Object(properties));
        schema.insert("required".to_string(), Value::Array(required));
        variants.push(Value::Object(schema));
    }

    let mut root = Map::new();
    root.insert("type".to_string(), json!("object"));
    root.insert(
        "description".to_string(),
        json!(
            "Output schema for the MCP orchestrator. Contains the parameters for the selected tool."
        ),
    );
    root.insert(
        "properties".to_string(),
        json!({
            "tool_parameters": {
                "description": "The parameters for the selected tool, matching its specific schema (which includes the 'tool_name').",
                "anyOf": variants,
            }
        }),
    );
    root.insert("required".to_string(), json!(["tool_parameters"]));
    if !root_defs.is_empty() {
        root.insert("$defs".to_string(), Value::Object(root_defs));
    }
    Ok(Value::Object(root))
}

/// A set of MCP tools with their orchestrator schema, ready to dispatch calls.
#[derive(Clone, Debug)]
pub struct McpToolset {
    tools: Vec<McpTool>,
    index: HashMap<String, usize>,
    schema: Value,
}

impl McpToolset {
    /// Builds the toolset and its schema. Errors under the same conditions as
    /// [`orchestrator_schema`].
    pub fn new(tools: Vec<McpTool>) -> Result<Self> {
        let schema = orchestrator_schema(&tools)?;
        let index = tools
            .iter()
            .enumerate()
            .map(|(i, t)| (t.mcp_name().to_string(), i))
            .collect();
        Ok(Self {
            tools,
            index,
            schema,
        })
    }

    /// The orchestrator schema, for `AgentConfig::with_output_schema`.
    pub fn schema(&self) -> Value {
        self.schema.clone()
    }

    /// The tool called `name`, if any.
    pub fn get(&self, name: &str) -> Option<&McpTool> {
        self.index.get(name).map(|&i| &self.tools[i])
    }

    /// The tools, in the order given to [`McpToolset::new`].
    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    /// Calls the tool the model selected with its arguments.
    ///
    /// Errors when the tool is not in this set (including a model-invented
    /// name) or when the call itself fails.
    pub async fn dispatch(&self, call: &McpToolCall) -> Result<Value> {
        let tool = self
            .get(&call.tool_name)
            .ok_or_else(|| Error::Mcp(format!("unknown MCP tool {:?}", call.tool_name)))?;
        tool.call(call.arguments.clone()).await
    }
}

/// Connects and lists the server's tools.
pub async fn fetch_mcp_tools(
    endpoint: &str,
    kind: McpTransportType,
) -> Result<(McpClient, Vec<McpTool>)> {
    let client = McpClient::connect_endpoint(endpoint, kind).await?;
    let tools = client.tools().await?;
    Ok((client, tools))
}

/// Connects, lists the server's tools, and builds their orchestrator schema.
pub async fn fetch_mcp_attributes_with_schema(
    endpoint: &str,
    kind: McpTransportType,
) -> Result<(McpClient, Vec<McpTool>, Value)> {
    let (client, tools) = fetch_mcp_tools(endpoint, kind).await?;
    let schema = orchestrator_schema(&tools)?;
    Ok((client, tools, schema))
}

/// Connects and lists the server's resources and resource templates.
pub async fn fetch_mcp_resources(
    endpoint: &str,
    kind: McpTransportType,
) -> Result<(McpClient, Vec<McpResource>)> {
    let client = McpClient::connect_endpoint(endpoint, kind).await?;
    let resources = client.resources().await?;
    Ok((client, resources))
}

/// Connects and lists the server's prompts.
pub async fn fetch_mcp_prompts(
    endpoint: &str,
    kind: McpTransportType,
) -> Result<(McpClient, Vec<McpPrompt>)> {
    let client = McpClient::connect_endpoint(endpoint, kind).await?;
    let prompts = client.prompts().await?;
    Ok((client, prompts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj() -> Value {
        json!({"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]})
    }

    #[test]
    fn two_tools_make_any_of() {
        let (s1, s2) = (obj(), json!({"type": "object"}));
        let schema =
            build_orchestrator_schema(&[("add", Some("Adds."), &s1), ("echo", None, &s2)]).unwrap();
        assert_eq!(schema["required"], json!(["tool_parameters"]));
        assert!(schema.get("$defs").is_none());
        let any_of = schema["properties"]["tool_parameters"]["anyOf"]
            .as_array()
            .unwrap();
        assert_eq!(any_of.len(), 2);
        for (variant, name) in any_of.iter().zip(["add", "echo"]) {
            let first = variant["properties"]
                .as_object()
                .unwrap()
                .iter()
                .next()
                .unwrap();
            assert_eq!(first.0, "tool_name");
            assert_eq!(first.1["enum"], json!([name]));
            assert_eq!(variant["required"][0], json!("tool_name"));
            assert_eq!(variant["title"], json!(name));
            assert_eq!(variant["type"], json!("object"));
        }
        assert_eq!(any_of[0]["required"], json!(["tool_name", "a"]));
        assert_eq!(any_of[0]["description"], json!("Adds."));
        assert!(any_of[1].get("description").is_none());
    }

    #[test]
    fn schema_is_deterministic() {
        let s = obj();
        let a = build_orchestrator_schema(&[("add", None, &s)]).unwrap();
        let b = build_orchestrator_schema(&[("add", None, &s)]).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    #[test]
    fn defs_are_hoisted_and_refs_rewritten() {
        for key in ["$defs", "definitions"] {
            let s = json!({
                "type": "object",
                "properties": {"p": {"$ref": format!("#/{key}/Pt")}},
                key: {"Pt": {"type": "object", "properties": {"q": {"$ref": format!("#/{key}/Inner/x")}}}, "Inner": {}},
            });
            let schema = build_orchestrator_schema(&[("geo-tool", None, &s)]).unwrap();
            assert!(schema["$defs"]["geo_tool__Pt"].is_object(), "{schema}");
            assert!(schema["$defs"]["geo_tool__Inner"].is_object());
            let variant = &schema["properties"]["tool_parameters"]["anyOf"][0];
            assert!(variant.get(key).is_none());
            assert_eq!(
                variant["properties"]["p"]["$ref"],
                json!("#/$defs/geo_tool__Pt")
            );
            assert_eq!(
                schema["$defs"]["geo_tool__Pt"]["properties"]["q"]["$ref"],
                json!("#/$defs/geo_tool__Inner/x")
            );
        }
    }

    #[test]
    fn tool_name_collision_errors() {
        let s = json!({"type": "object", "properties": {"tool_name": {"type": "string"}}});
        let err = build_orchestrator_schema(&[("t", None, &s)]).unwrap_err();
        assert!(err.to_string().contains("tool_name"), "{err}");
    }

    #[test]
    fn empty_and_duplicate_error() {
        let err = build_orchestrator_schema(&[]).unwrap_err();
        assert!(err.to_string().contains("at least one tool"), "{err}");
        let s = obj();
        let err = build_orchestrator_schema(&[("t", None, &s), ("t", None, &s)]).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "{err}");
    }

    #[test]
    fn non_object_schema_is_tolerated() {
        let s = Value::Bool(true);
        let schema = build_orchestrator_schema(&[("t", None, &s)]).unwrap();
        let variant = &schema["properties"]["tool_parameters"]["anyOf"][0];
        assert_eq!(variant["type"], json!("object"));
        assert_eq!(variant["required"], json!(["tool_name"]));
        assert!(variant["properties"]["tool_name"].is_object());
    }

    #[test]
    fn tool_call_round_trips() {
        let call: McpToolCall =
            serde_json::from_value(json!({"tool_name": "add", "a": 1})).unwrap();
        assert_eq!(call.tool_name, "add");
        assert_eq!(Value::Object(call.arguments.clone()), json!({"a": 1}));
        assert_eq!(
            serde_json::to_value(&call).unwrap(),
            json!({"tool_name": "add", "a": 1})
        );
        assert_eq!(McpToolCall::schema_title(), "McpToolCall");
        assert_eq!(
            McpOrchestratorOutput::schema_title(),
            "McpOrchestratorOutput"
        );
    }
}
