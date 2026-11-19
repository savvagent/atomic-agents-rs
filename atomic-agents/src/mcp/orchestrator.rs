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
//! Each tool's schema pins `tool_name` to the tool via an `enum` and lists it
//! first in `required`. Property order carries no meaning in JSON Schema.
//!
//! Each tool's own `$schema` keyword is dropped, and its `$defs` /
//! `definitions` are hoisted into the root `$defs` with refs rewritten to
//! match. Gemini function declarations do not support `$ref` or `$defs`, so
//! with Google models a tool whose input schema uses them may need its
//! schema flattened (refs inlined) first.

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
/// See the module docs. The output is deterministic: object keys are sorted
/// (`serde_json`'s default map), while `anyOf` follows the order of `tools` and
/// `required` keeps `tool_name` first. The `serde_json/preserve_order` feature
/// is deliberately not enabled, because feature unification would change map
/// ordering for every crate in a downstream dependency graph.
///
/// Gemini does not support `$ref`/`$defs` in function declarations; see the
/// module docs.
///
/// Errors when `tools` is empty, two tools share a name,
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

/// Escapes a string for use as one JSON-pointer reference token.
fn escape_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

/// Reverses [`escape_token`].
fn unescape_token(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

/// Rewrites refs into one tool's own `$defs`/`definitions` so they point at the
/// hoisted `{prefix}__{def}` entries of the root `$defs`.
struct RefRewriter<'a> {
    tool: &'a str,
    prefix: &'a str,
    defs: &'a std::collections::HashSet<String>,
}

impl RefRewriter<'_> {
    fn unsupported(&self, reference: &str) -> Error {
        Error::Mcp(format!(
            "tool {:?}: unsupported $ref {reference:?} (only refs into the tool's own $defs/definitions are supported)",
            self.tool
        ))
    }

    fn rewrite_ref(&self, reference: &str) -> Result<Option<String>> {
        if !reference.starts_with('#') {
            // Absolute or relative URI: not ours to touch.
            return Ok(None);
        }
        let rest = reference
            .strip_prefix("#/$defs/")
            .or_else(|| reference.strip_prefix("#/definitions/"))
            .ok_or_else(|| self.unsupported(reference))?;
        let (token, suffix) = match rest.find('/') {
            Some(i) => rest.split_at(i),
            None => (rest, ""),
        };
        let name = unescape_token(token);
        if !self.defs.contains(&name) {
            return Err(self.unsupported(reference));
        }
        let key = format!("{}__{name}", self.prefix);
        Ok(Some(format!("#/$defs/{}{suffix}", escape_token(&key))))
    }

    /// Walks a schema object. Keywords whose values are instance data are
    /// skipped, and maps keyed by user-chosen names have each value walked as
    /// a schema, so a property called `default` is still rewritten.
    fn walk_schema(&self, schema: &mut Map<String, Value>) -> Result<()> {
        for (key, value) in schema.iter_mut() {
            match key.as_str() {
                "enum" | "const" | "default" | "examples" => {}
                "$ref" => {
                    if let Value::String(r) = &*value
                        && let Some(new) = self.rewrite_ref(r)?
                    {
                        *value = Value::String(new);
                    }
                }
                "properties" | "patternProperties" | "$defs" | "definitions"
                | "dependentSchemas" => {
                    if let Value::Object(children) = value {
                        for child in children.values_mut() {
                            self.walk_value(child)?;
                        }
                    }
                }
                _ => self.walk_value(value)?,
            }
        }
        Ok(())
    }

    fn walk_value(&self, value: &mut Value) -> Result<()> {
        match value {
            Value::Object(map) => self.walk_schema(map),
            Value::Array(items) => items.iter_mut().try_for_each(|v| self.walk_value(v)),
            _ => Ok(()),
        }
    }
}

/// Moves `$defs` and `definitions` out of `schema` into one map, refusing a
/// name present in both.
fn take_defs(tool: &str, schema: &mut Map<String, Value>) -> Result<Map<String, Value>> {
    let mut defs = Map::new();
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(found)) = schema.remove(key) {
            for (name, def) in found {
                if defs.insert(name.clone(), def).is_some() {
                    return Err(Error::Mcp(format!(
                        "tool {tool:?}: definition {name:?} appears in both $defs and definitions"
                    )));
                }
            }
        }
    }
    Ok(defs)
}

/// The arguments to send to the tool: everything except `tool_name`.
fn forwarded_arguments(call: &McpToolCall) -> Map<String, Value> {
    let mut arguments = call.arguments.clone();
    arguments.remove("tool_name");
    arguments
}

/// Pure schema construction over `(name, description, input_schema)` triples.
pub(crate) fn build_orchestrator_schema(entries: &[(&str, Option<&str>, &Value)]) -> Result<Value> {
    if entries.is_empty() {
        return Err(Error::Mcp(
            "orchestrator schema needs at least one tool".to_string(),
        ));
    }
    let mut names = std::collections::HashSet::new();
    // sanitized prefix -> (first tool using it, whether that tool has defs)
    let mut prefixes: HashMap<String, (&str, bool)> = HashMap::new();
    let mut variants = Vec::with_capacity(entries.len());
    let mut root_defs = Map::new();

    for &(name, description, input_schema) in entries {
        if !names.insert(name) {
            return Err(Error::Mcp(format!("duplicate MCP tool name {name:?}")));
        }
        // A `true` (or otherwise non-object) schema accepts anything: no arguments declared.
        let mut schema = match input_schema {
            Value::Object(map) => map.clone(),
            _ => Map::new(),
        };
        if let Some(ty) = schema.get("type")
            && ty != "object"
        {
            return Err(Error::Mcp(format!(
                "tool {name:?}: input schema type must be \"object\", got {ty}"
            )));
        }
        // A dialect declaration is only meaningful at a document root; inside
        // the `anyOf` it is noise, and some providers reject it.
        schema.remove("$schema");
        let mut defs = take_defs(name, &mut schema)?;
        let prefix = sanitize(name);
        let has_defs = !defs.is_empty();
        if let Some(&(other, other_has_defs)) = prefixes.get(&prefix) {
            if has_defs || other_has_defs {
                return Err(Error::Mcp(format!(
                    "tools {other:?} and {name:?} both map to schema prefix {prefix:?}; rename one"
                )));
            }
        } else {
            prefixes.insert(prefix.clone(), (name, has_defs));
        }

        let mut properties = match schema.remove("properties") {
            Some(Value::Object(p)) => p,
            _ => Map::new(),
        };
        if properties.contains_key("tool_name") {
            return Err(Error::Mcp(format!(
                "MCP tool {name:?} declares a `tool_name` property, which the orchestrator reserves"
            )));
        }

        // Rewrite refs everywhere the tool's own schema can hold one.
        let def_names: std::collections::HashSet<String> = defs.keys().cloned().collect();
        let rewriter = RefRewriter {
            tool: name,
            prefix: &prefix,
            defs: &def_names,
        };
        rewriter.walk_schema(&mut schema)?;
        for property in properties.values_mut() {
            rewriter.walk_value(property)?;
        }
        for def in defs.values_mut() {
            rewriter.walk_value(def)?;
        }
        for (def_name, def) in std::mem::take(&mut defs) {
            let key = format!("{prefix}__{def_name}");
            if root_defs.insert(key.clone(), def).is_some() {
                return Err(Error::Mcp(format!(
                    "tool {name:?}: definition key {key:?} collides with another tool's definition"
                )));
            }
        }

        properties.insert(
            "tool_name".to_string(),
            json!({
                "type": "string",
                "enum": [name],
                "description": format!("Required identifier for the {name} tool."),
            }),
        );

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
        let tool = self.get(&call.tool_name).ok_or_else(|| {
            let available: Vec<&str> = self.tools.iter().map(|t| t.mcp_name()).collect();
            Error::Mcp(format!(
                "unknown MCP tool {:?}; available: {}",
                call.tool_name,
                available.join(", ")
            ))
        })?;
        tool.call(forwarded_arguments(call)).await
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
            assert_eq!(variant["properties"]["tool_name"]["enum"], json!([name]));
            assert_eq!(variant["required"][0], json!("tool_name"));
            assert_eq!(variant["title"], json!(name));
            assert_eq!(variant["type"], json!("object"));
        }
        assert_eq!(any_of[0]["required"], json!(["tool_name", "a"]));
        assert_eq!(any_of[0]["description"], json!("Adds."));
        assert!(any_of[1].get("description").is_none());
    }

    #[test]
    fn member_schema_keywords_are_stripped() {
        let with_dialect = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"a": {"type": "number"}}
        });
        let schema = build_orchestrator_schema(&[("add", None, &with_dialect)]).unwrap();
        let member = &schema["properties"]["tool_parameters"]["anyOf"][0];
        assert!(member.get("$schema").is_none(), "{member}");
        assert_eq!(member["properties"]["a"], json!({"type": "number"}));
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

    fn err_of(entries: &[(&str, Option<&str>, &Value)]) -> String {
        build_orchestrator_schema(entries).unwrap_err().to_string()
    }

    #[test]
    fn colliding_prefixes_with_defs_error() {
        let with_defs = json!({"type": "object", "$defs": {"P": {"type": "string"}}});
        let plain = obj();
        let msg = err_of(&[("a-b", None, &with_defs), ("a_b", None, &plain)]);
        assert!(
            msg.contains(r#"tools "a-b" and "a_b" both map to schema prefix "a_b"; rename one"#),
            "{msg}"
        );
        // Either order, and defs on the second tool.
        assert!(err_of(&[("a_b", None, &plain), ("a-b", None, &with_defs)]).contains("prefix"));
        // Colliding prefixes without any defs are harmless.
        build_orchestrator_schema(&[("a-b", None, &plain), ("a_b", None, &plain)]).unwrap();
    }

    #[test]
    fn same_def_in_defs_and_definitions_errors() {
        let s = json!({"$defs": {"P": {}}, "definitions": {"P": {}}});
        let msg = err_of(&[("t", None, &s)]);
        assert!(msg.contains("P") && msg.contains("both"), "{msg}");
    }

    #[test]
    fn escaped_def_names_are_resolved_and_reescaped() {
        let s = json!({
            "properties": {
                "a": {"$ref": "#/$defs/x~1y"},
                "b": {"$ref": "#/definitions/m~0n/properties/z"},
            },
            "$defs": {"x/y": {"type": "string"}},
            "definitions": {"m~n": {"type": "object"}},
        });
        let schema = build_orchestrator_schema(&[("t", None, &s)]).unwrap();
        let props = &schema["properties"]["tool_parameters"]["anyOf"][0]["properties"];
        assert_eq!(props["a"]["$ref"], json!("#/$defs/t__x~1y"));
        assert_eq!(props["b"]["$ref"], json!("#/$defs/t__m~0n/properties/z"));
        assert!(schema["$defs"]["t__x/y"].is_object());
        assert!(schema["$defs"]["t__m~n"].is_object());
    }

    #[test]
    fn unsupported_refs_error() {
        let cases = [
            ("#", json!({"properties": {"x": {"$ref": "#"}}})),
            (
                "#/properties/x",
                json!({"properties": {"y": {"$ref": "#/properties/x"}}}),
            ),
            (
                "#/$defs/Missing",
                json!({"properties": {"y": {"$ref": "#/$defs/Missing"}}, "$defs": {"Other": {}}}),
            ),
            (
                "#/$defs/Missing",
                json!({"properties": {"y": {"$ref": "#/$defs/Missing"}}}),
            ),
            // A ref inside a hoisted def is checked too.
            (
                "#/$defs/Nope",
                json!({"$defs": {"P": {"items": {"$ref": "#/$defs/Nope"}}}}),
            ),
        ];
        for (r, schema) in cases {
            let msg = err_of(&[("t", None, &schema)]);
            assert_eq!(
                msg.trim_start_matches("MCP error: "),
                format!(
                    "tool \"t\": unsupported $ref \"{r}\" (only refs into the tool's own $defs/definitions are supported)"
                )
            );
        }
    }

    #[test]
    fn absolute_refs_and_data_positions_are_untouched() {
        let s = json!({
            "properties": {
                "u": {"$ref": "https://example.com/schema.json"},
                "d": {"default": {"$ref": "#/nowhere"}, "const": {"$ref": "#"},
                      "enum": [{"$ref": "#/x"}], "examples": [{"$ref": "#/y"}]},
                "default": {"$ref": "#/$defs/P"},
            },
            "$defs": {"P": {}},
        });
        let schema = build_orchestrator_schema(&[("t", None, &s)]).unwrap();
        let props = &schema["properties"]["tool_parameters"]["anyOf"][0]["properties"];
        assert_eq!(props["u"]["$ref"], json!("https://example.com/schema.json"));
        assert_eq!(props["d"]["default"], json!({"$ref": "#/nowhere"}));
        assert_eq!(props["d"]["const"], json!({"$ref": "#"}));
        assert_eq!(props["d"]["enum"], json!([{"$ref": "#/x"}]));
        assert_eq!(props["d"]["examples"], json!([{"$ref": "#/y"}]));
        // A property that happens to be named `default` is still a schema.
        assert_eq!(props["default"]["$ref"], json!("#/$defs/t__P"));
    }

    #[test]
    fn non_object_type_errors_but_true_schema_is_empty() {
        let s = json!({"type": "string"});
        assert!(err_of(&[("t", None, &s)]).contains("object"));
        let t = Value::Bool(true);
        build_orchestrator_schema(&[("t", None, &t)]).unwrap();
    }

    #[test]
    fn forwarded_arguments_exclude_tool_name() {
        let mut arguments = Map::new();
        arguments.insert("a".into(), json!(1));
        arguments.insert("tool_name".into(), json!("add"));
        let call = McpToolCall {
            tool_name: "add".into(),
            arguments,
        };
        assert_eq!(Value::Object(forwarded_arguments(&call)), json!({"a": 1}));
        let parsed: McpToolCall =
            serde_json::from_value(json!({"tool_name": "add", "a": 1})).unwrap();
        assert_eq!(Value::Object(forwarded_arguments(&parsed)), json!({"a": 1}));
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
