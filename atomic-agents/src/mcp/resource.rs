//! [`McpResource`]: a discovered MCP resource or resource template exposed
//! through the framework's [`Resource`] trait.

use rmcp::model::ResourceContents;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::client::McpClient;
use super::definitions::McpResourceDefinition;
use super::tool::{McpToolInput, block_on_multi_thread};
use crate::error::{Error, Result};
use crate::resource::Resource;
use crate::schema::IOSchema;
use crate::tool::ComponentConfig;

/// One content block returned by `resources/read`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpResourceContent {
    /// UTF-8 text content.
    Text {
        /// URI the content was read from.
        uri: String,
        /// MIME type, if the server sent one.
        mime_type: Option<String>,
        /// The text.
        text: String,
    },
    /// Binary content.
    Blob {
        /// URI the content was read from.
        uri: String,
        /// MIME type, if the server sent one.
        mime_type: Option<String>,
        /// Base64-encoded bytes, exactly as the server sent them.
        blob: String,
    },
}

impl TryFrom<ResourceContents> for McpResourceContent {
    type Error = Error;

    fn try_from(contents: ResourceContents) -> Result<Self> {
        Ok(match contents {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => Self::Text {
                uri,
                mime_type,
                text,
            },
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => Self::Blob {
                uri,
                mime_type,
                blob,
            },
            _ => {
                return Err(Error::Mcp(
                    "resources/read returned an unsupported content type".to_string(),
                ));
            }
        })
    }
}

/// Result of reading an MCP resource.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct McpResourceOutput {
    /// Every content block the server returned, in order.
    pub contents: Vec<McpResourceContent>,
    /// MIME type of the first content block, if it has one.
    pub mime_type: Option<String>,
}

impl McpResourceOutput {
    pub(crate) fn from_contents(contents: Vec<ResourceContents>) -> Result<Self> {
        let contents = contents
            .into_iter()
            .map(McpResourceContent::try_from)
            .collect::<Result<Vec<_>>>()?;
        let mime_type = contents.first().and_then(|c| match c {
            McpResourceContent::Text { mime_type, .. }
            | McpResourceContent::Blob { mime_type, .. } => mime_type.clone(),
        });
        Ok(Self {
            contents,
            mime_type,
        })
    }
}

impl IOSchema for McpResourceOutput {
    fn schema_title() -> &'static str {
        "McpResourceOutput"
    }
    fn schema_description() -> &'static str {
        "Contents of an MCP resource."
    }
}

/// An MCP resource or resource template bound to the session it was
/// discovered on.
///
/// Implements [`Resource`] with the template parameters as input.
/// [`Resource::read`] is synchronous, so it blocks the calling thread and
/// requires a multi-threaded tokio runtime; in async code use
/// [`McpResource::read_async`].
#[derive(Clone, Debug)]
pub struct McpResource {
    definition: McpResourceDefinition,
    client: McpClient,
    config: ComponentConfig,
}

impl McpResource {
    /// Binds `definition` to the session it was discovered on.
    pub fn from_definition(definition: McpResourceDefinition, client: McpClient) -> Self {
        let description = definition
            .description
            .clone()
            .unwrap_or_else(|| format!("MCP resource {}", definition.name));
        let config = ComponentConfig::new()
            .with_title(definition.name.clone())
            .with_description(description);
        Self {
            definition,
            client,
            config,
        }
    }

    /// The server's definition of this resource.
    pub fn definition(&self) -> &McpResourceDefinition {
        &self.definition
    }

    /// The URI to read for `args`: for a template, the template expanded with
    /// `args`; for a concrete resource, its URI unchanged (`args` are ignored
    /// and the URI is never brace-parsed).
    pub fn uri_for(&self, args: &Map<String, Value>) -> Result<String> {
        uri_for_definition(&self.definition, args)
    }

    /// Reads the resource asynchronously.
    pub async fn read_async(&self, args: Map<String, Value>) -> Result<McpResourceOutput> {
        let uri = self.uri_for(&args)?;
        self.client.read_resource(&uri).await
    }
}

impl Resource for McpResource {
    type Input = McpToolInput;
    type Output = McpResourceOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn read(&self, input: Self::Input) -> Result<Self::Output> {
        block_on_multi_thread(
            self.read_async(input.0),
            "McpResource::read",
            "McpResource::read_async",
        )
        .and_then(|r| r)
    }
}

fn uri_for_definition(def: &McpResourceDefinition, args: &Map<String, Value>) -> Result<String> {
    if def.is_template {
        fill_template(&def.uri, args)
    } else {
        Ok(def.uri.clone())
    }
}

const RESERVED: &str = ":/?#[]@!$&'()*+,;=";

/// Percent-encodes `s`, keeping unreserved characters (and, when
/// `keep_reserved`, RFC 3986 reserved ones).
fn encode(s: &str, keep_reserved: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric()
            || matches!(c, '-' | '.' | '_' | '~')
            || (keep_reserved && RESERVED.contains(c))
        {
            out.push(c);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn arg_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Expands an RFC 6570 URI template with `args`.
///
/// Supports the simple, `+`, `#`, `.`, `/`, `;`, `?` and `&` operators.
/// Values are percent-encoded except unreserved characters (and reserved
/// ones for `+` / `#`); non-string values use their JSON text. A missing
/// variable is an error, except for `?` / `&` (form-style query) variables,
/// which are omitted. Prefix (`:n`) and explode (`*`) modifiers are ignored.
pub(crate) fn fill_template(template: &str, args: &Map<String, Value>) -> Result<String> {
    let mut out = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else { break };
        out.push_str(&rest[..open]);
        expand(&after[..close], args, &mut out)?;
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn expand(expr: &str, args: &Map<String, Value>, out: &mut String) -> Result<()> {
    let (op, list) = match expr.chars().next() {
        Some(c @ ('+' | '#' | '.' | '/' | ';' | '?' | '&')) => (Some(c), &expr[1..]),
        _ => (None, expr),
    };
    let optional = matches!(op, Some('?' | '&'));
    let keep_reserved = matches!(op, Some('+' | '#'));
    let mut parts = Vec::new();
    for spec in list.split(',') {
        let name = spec
            .split(':')
            .next()
            .unwrap_or("")
            .trim_end_matches('*')
            .trim();
        if name.is_empty() {
            continue;
        }
        let Some(value) = args.get(name).filter(|v| !v.is_null()) else {
            if optional {
                continue;
            }
            return Err(Error::Mcp(format!(
                "missing required resource template parameter '{name}'"
            )));
        };
        let value = encode(&arg_text(value), keep_reserved);
        parts.push(match op {
            Some('?' | '&') => format!("{name}={value}"),
            Some(';') if value.is_empty() => name.to_string(),
            Some(';') => format!("{name}={value}"),
            _ => value,
        });
    }
    if parts.is_empty() {
        return Ok(());
    }
    let (prefix, sep) = match op {
        Some('#') => ("#", ","),
        Some('.') => (".", "."),
        Some('/') => ("/", "/"),
        Some(';') => (";", ";"),
        Some('?') => ("?", "&"),
        Some('&') => ("&", "&"),
        _ => ("", ","),
    };
    out.push_str(prefix);
    out.push_str(&parts.join(sep));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn fills_simple_placeholder() {
        assert_eq!(
            fill_template("mem://notes/{id}", &args(json!({"id": "7"}))).unwrap(),
            "mem://notes/7"
        );
    }

    #[test]
    fn simple_values_are_fully_percent_encoded() {
        assert_eq!(
            fill_template("mem://notes/{id}", &args(json!({"id": "a b/c"}))).unwrap(),
            "mem://notes/a%20b%2Fc"
        );
        assert_eq!(
            fill_template("x/{id}", &args(json!({"id": "A-z0._~"}))).unwrap(),
            "x/A-z0._~"
        );
    }

    #[test]
    fn plus_keeps_reserved_characters() {
        let all = "/:?#[]@!$&'()*+,;=";
        assert_eq!(
            fill_template("x/{+p}", &args(json!({"p": format!("{all} b")}))).unwrap(),
            format!("x/{all}%20b")
        );
    }

    #[test]
    fn missing_required_names_the_parameter() {
        for t in ["x/{id}", "x/{+id}", "x/{#id}"] {
            let err = fill_template(t, &Map::new()).unwrap_err().to_string();
            assert!(err.contains("'id'"), "{t}: {err}");
        }
    }

    #[test]
    fn non_string_values_use_json_text() {
        assert_eq!(
            fill_template("x/{id}", &args(json!({"id": 7}))).unwrap(),
            "x/7"
        );
        assert_eq!(
            fill_template("x/{id}", &args(json!({"id": true}))).unwrap(),
            "x/true"
        );
    }

    #[test]
    fn query_expansion_omits_missing() {
        let t = "x{?a,b}";
        assert_eq!(
            fill_template(t, &args(json!({"a": 1, "b": 2}))).unwrap(),
            "x?a=1&b=2"
        );
        assert_eq!(
            fill_template(t, &args(json!({"b": "z y"}))).unwrap(),
            "x?b=z%20y"
        );
        assert_eq!(fill_template(t, &Map::new()).unwrap(), "x");
        assert_eq!(
            fill_template("x?k=1{&a}", &args(json!({"a": 2}))).unwrap(),
            "x?k=1&a=2"
        );
    }

    #[test]
    fn other_operators() {
        let a = args(json!({"v": "a b", "w": "c"}));
        assert_eq!(
            fill_template("x{#v}", &a).unwrap(),
            "x#a b".replace(' ', "%20")
        );
        assert_eq!(fill_template("x{/v,w}", &a).unwrap(), "x/a%20b/c");
        assert_eq!(fill_template("x{.w}", &a).unwrap(), "x.c");
        assert_eq!(fill_template("x{;w}", &a).unwrap(), "x;w=c");
        assert_eq!(fill_template("x{v,w}", &a).unwrap(), "xa%20b,c");
    }

    #[test]
    fn content_conversion() {
        let text: McpResourceContent = ResourceContents::text("hi", "mem://a")
            .with_mime_type("text/x")
            .try_into()
            .unwrap();
        assert_eq!(
            text,
            McpResourceContent::Text {
                uri: "mem://a".into(),
                mime_type: Some("text/x".into()),
                text: "hi".into()
            }
        );
        let blob: McpResourceContent = ResourceContents::blob("AAEC", "mem://b")
            .try_into()
            .unwrap();
        assert_eq!(
            serde_json::to_value(&blob).unwrap(),
            json!({"type": "blob", "uri": "mem://b", "mime_type": null, "blob": "AAEC"})
        );
    }

    #[test]
    fn output_mime_is_first_contents() {
        let out = McpResourceOutput::from_contents(vec![
            ResourceContents::blob("AA", "mem://b").with_mime_type("image/png"),
            ResourceContents::text("t", "mem://t"),
        ])
        .unwrap();
        assert_eq!(out.mime_type.as_deref(), Some("image/png"));
        assert_eq!(out.contents.len(), 2);
        let empty = McpResourceOutput::from_contents(vec![]).unwrap();
        assert_eq!(empty.mime_type, None);
        assert_eq!(McpResourceOutput::schema_title(), "McpResourceOutput");
    }

    #[test]
    fn concrete_uri_is_never_parsed_or_rewritten() {
        for uri in ["mem://x/%7Ba%7D", "mem://x/{not-template}"] {
            let def = McpResourceDefinition::from_resource(rmcp::model::Resource::new(uri, "x"));
            let got = uri_for_definition(&def, &args(json!({"a": "1"}))).unwrap();
            assert_eq!(got, uri);
        }
    }
}
