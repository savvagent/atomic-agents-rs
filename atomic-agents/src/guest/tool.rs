//! The tool side of the guest: [`McpHost`], [`HostTool`] and [`HostToolset`].

use std::collections::HashMap;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::llm::ToolSpec;
use crate::error::{Error, Result as AgentResult};
use crate::schema::IOSchema;
use crate::tool::{ComponentConfig, Tool};

/// A tool the host offers, as listed by [`McpHost::list_tools`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Tool name, as passed to [`McpHost::call`].
    pub name: String,
    /// What the tool does.
    pub description: String,
    /// JSON Schema of the arguments, as a JSON document.
    pub input_schema: String,
}

impl From<ToolInfo> for ToolSpec {
    fn from(info: ToolInfo) -> Self {
        Self {
            name: info.name,
            description: info.description,
            input_schema: info.input_schema,
        }
    }
}

/// Why a host tool call or listing failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolError {
    /// The host does not permit this call.
    Denied(String),
    /// No such tool.
    NotFound(String),
    /// The arguments were rejected.
    InvalidArguments(String),
    /// The tool ran and failed.
    Failed(String),
    /// A quota or limit was reached.
    LimitExceeded(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Denied(m) => write!(f, "denied: {m}"),
            Self::NotFound(m) => write!(f, "not found: {m}"),
            Self::InvalidArguments(m) => write!(f, "invalid arguments: {m}"),
            Self::Failed(m) => write!(f, "failed: {m}"),
            Self::LimitExceeded(m) => write!(f, "limit exceeded: {m}"),
        }
    }
}

impl std::error::Error for ToolError {}

impl From<ToolError> for Error {
    fn from(e: ToolError) -> Self {
        Error::Mcp(format!("host mcp: {e}"))
    }
}

/// The tools the host provides to the guest.
///
/// The calls are synchronous. Implemented for `&T` and `Arc<T>` whenever `T`
/// implements it.
pub trait McpHost: Send + Sync {
    /// Calls `tool` with `args`, a JSON document, and returns its output.
    fn call(&self, tool: &str, args: &str) -> Result<String, ToolError>;

    /// Lists the tools the host offers.
    fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolError>;
}

impl<T: McpHost + ?Sized> McpHost for &T {
    fn call(&self, tool: &str, args: &str) -> Result<String, ToolError> {
        (**self).call(tool, args)
    }

    fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolError> {
        (**self).list_tools()
    }
}

impl<T: McpHost + ?Sized> McpHost for Arc<T> {
    fn call(&self, tool: &str, args: &str) -> Result<String, ToolError> {
        (**self).call(tool, args)
    }

    fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolError> {
        (**self).list_tools()
    }
}

/// Arguments for a host tool call, as a JSON object.
///
/// Serializes transparently as the object itself.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct HostToolInput(pub Map<String, Value>);

impl IOSchema for HostToolInput {
    fn schema_title() -> &'static str {
        "HostToolInput"
    }
    fn schema_description() -> &'static str {
        "Arguments for a host tool call."
    }
}

/// Result of a host tool call: the output parsed as JSON, or the raw text as
/// a JSON string when it is not JSON.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct HostToolOutput {
    /// The converted tool result.
    pub result: Value,
}

impl IOSchema for HostToolOutput {
    fn schema_title() -> &'static str {
        "HostToolOutput"
    }
    fn schema_description() -> &'static str {
        "Result of a host tool call."
    }
}

/// A host tool exposed through the framework's [`Tool`] trait.
///
/// [`Tool::run`] and [`HostTool::call`] block on the host call.
#[derive(Clone, Debug)]
pub struct HostTool<H: McpHost> {
    host: H,
    info: ToolInfo,
    schema: Value,
    config: ComponentConfig,
}

impl<H: McpHost> HostTool<H> {
    /// Binds `info` to the host it was listed from.
    ///
    /// Errors with [`Error::Mcp`] when `info.input_schema` is not valid JSON,
    /// so [`HostTool::input_schema`] and the [`ToolSpec`] sent to the model
    /// always agree.
    pub fn new(host: H, info: ToolInfo) -> AgentResult<Self> {
        let schema = serde_json::from_str(&info.input_schema).map_err(|e| {
            Error::Mcp(format!(
                "host tool {:?} has an invalid input schema: {e}",
                info.name
            ))
        })?;
        let config = ComponentConfig::new()
            .with_title(info.name.clone())
            .with_description(info.description.clone());
        Ok(Self {
            host,
            info,
            schema,
            config,
        })
    }

    /// The tool name used in [`McpHost::call`].
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// JSON Schema of the arguments, as the host sent it.
    pub fn input_schema(&self) -> Value {
        self.schema.clone()
    }

    /// Calls the tool. The output is parsed as JSON; when it is not JSON it
    /// is returned as a [`Value::String`].
    pub fn call(&self, args: Map<String, Value>) -> AgentResult<Value> {
        let args = serde_json::to_string(&args)?;
        let out = self.host.call(&self.info.name, &args)?;
        Ok(serde_json::from_str(&out).unwrap_or(Value::String(out)))
    }
}

impl<H: McpHost> Tool for HostTool<H> {
    type Input = HostToolInput;
    type Output = HostToolOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> AgentResult<Self::Output> {
        self.call(input.0).map(|result| HostToolOutput { result })
    }
}

/// The host's tools, by name, with dispatch.
///
/// ```
/// use atomic_agents::guest::{HostToolset, McpHost, ToolError, ToolInfo};
///
/// struct Host;
/// impl McpHost for Host {
///     fn call(&self, _tool: &str, _args: &str) -> Result<String, ToolError> {
///         Ok(r#"{"ok":true}"#.into())
///     }
///     fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolError> {
///         Ok(vec![ToolInfo {
///             name: "ping".into(),
///             description: "Ping".into(),
///             input_schema: "{}".into(),
///         }])
///     }
/// }
///
/// # fn main() -> atomic_agents::error::Result<()> {
/// let toolset = HostToolset::from_host(std::sync::Arc::new(Host))?;
/// let out = toolset.dispatch("ping", serde_json::Map::new())?;
/// assert_eq!(out["ok"], true);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct HostToolset<H: McpHost + Clone> {
    tools: Vec<HostTool<H>>,
    index: HashMap<String, usize>,
}

impl<H: McpHost + Clone> HostToolset<H> {
    /// Lists the host's tools and binds each to a clone of `host`.
    pub fn from_host(host: H) -> AgentResult<Self> {
        let tools = host
            .list_tools()?
            .into_iter()
            .map(|info| HostTool::new(host.clone(), info))
            .collect::<AgentResult<Vec<_>>>()?;
        Ok(Self::new(tools))
    }

    /// A toolset over `tools`. A later tool replaces an earlier one of the
    /// same name in [`HostToolset::get`] and [`HostToolset::dispatch`].
    pub fn new(tools: Vec<HostTool<H>>) -> Self {
        let index = tools
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name().to_string(), i))
            .collect();
        Self { tools, index }
    }

    /// The tool called `name`, if any.
    pub fn get(&self, name: &str) -> Option<&HostTool<H>> {
        self.index.get(name).map(|&i| &self.tools[i])
    }

    /// The tools, in the order given.
    pub fn tools(&self) -> &[HostTool<H>] {
        &self.tools
    }

    /// The tools as [`ToolSpec`]s, for a completion request.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.info.clone().into()).collect()
    }

    /// Calls the tool the model selected.
    ///
    /// Errors when the tool is not in this set or when the call fails.
    pub fn dispatch(&self, name: &str, args: Map<String, Value>) -> AgentResult<Value> {
        let tool = self.get(name).ok_or_else(|| {
            let available: Vec<&str> = self.tools.iter().map(|t| t.name()).collect();
            Error::Mcp(format!(
                "unknown host tool {name:?}; available: {}",
                available.join(", ")
            ))
        })?;
        tool.call(args)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::{Map, Value, json};

    use super::*;
    use crate::error::Error;
    use crate::tool::Tool;

    #[derive(Default)]
    struct MockMcp {
        calls: Mutex<Vec<(String, String)>>,
        list_error: Option<ToolError>,
        bad_schema: bool,
    }

    impl McpHost for MockMcp {
        fn call(&self, tool: &str, args: &str) -> Result<String, ToolError> {
            self.calls
                .lock()
                .unwrap()
                .push((tool.to_string(), args.to_string()));
            match tool {
                "json" => Ok(r#"{"n":1}"#.to_string()),
                "plain" => Ok("just words".to_string()),
                "denied" => Err(ToolError::Denied("no".to_string())),
                _ => Err(ToolError::NotFound(tool.to_string())),
            }
        }

        fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolError> {
            if let Some(ToolError::Failed(m)) = &self.list_error {
                return Err(ToolError::Failed(m.clone()));
            }
            Ok(["json", "plain", "denied"]
                .iter()
                .map(|n| ToolInfo {
                    name: n.to_string(),
                    description: format!("the {n} tool"),
                    input_schema: if self.bad_schema && *n == "plain" {
                        "{not json".to_string()
                    } else {
                        r#"{"type":"object"}"#.to_string()
                    },
                })
                .collect())
        }
    }

    fn args() -> Map<String, Value> {
        json!({"q": "x"}).as_object().unwrap().clone()
    }

    #[test]
    fn from_host_lists_tools_and_exposes_specs() {
        let toolset = HostToolset::from_host(Arc::new(MockMcp::default())).unwrap();
        assert_eq!(toolset.tools().len(), 3);
        assert!(toolset.get("json").is_some());
        assert!(toolset.get("missing").is_none());
        let specs = toolset.tool_specs();
        assert_eq!(specs[0].name, "json");
        assert_eq!(specs[0].description, "the json tool");
        assert_eq!(specs[0].input_schema, r#"{"type":"object"}"#);
    }

    #[test]
    fn dispatch_parses_json_and_falls_back_to_string() {
        let host = Arc::new(MockMcp::default());
        let toolset = HostToolset::from_host(host.clone()).unwrap();
        assert_eq!(toolset.dispatch("json", args()).unwrap(), json!({"n": 1}));
        assert_eq!(
            toolset.dispatch("plain", args()).unwrap(),
            json!("just words")
        );
        let calls = host.calls.lock().unwrap();
        assert_eq!(calls[0].0, "json");
        let sent: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(sent, json!({"q": "x"}));
    }

    #[test]
    fn dispatch_unknown_tool_is_an_mcp_error() {
        let toolset = HostToolset::from_host(Arc::new(MockMcp::default())).unwrap();
        match toolset.dispatch("nope", args()) {
            Err(Error::Mcp(m)) => assert!(m.contains("nope"), "{m}"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn host_errors_map_to_mcp_errors() {
        let toolset = HostToolset::from_host(Arc::new(MockMcp::default())).unwrap();
        match toolset.dispatch("denied", args()) {
            Err(Error::Mcp(m)) => assert_eq!(m, "host mcp: denied: no"),
            other => panic!("unexpected {other:?}"),
        }
        let failing = MockMcp {
            list_error: Some(ToolError::Failed("down".into())),
            ..Default::default()
        };
        match HostToolset::from_host(Arc::new(failing)) {
            Err(Error::Mcp(m)) => assert_eq!(m, "host mcp: failed: down"),
            other => panic!("unexpected {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn invalid_input_schema_is_rejected_when_listing() {
        let host = Arc::new(MockMcp {
            bad_schema: true,
            ..Default::default()
        });
        match HostToolset::from_host(host) {
            Err(Error::Mcp(m)) => assert!(m.contains("plain"), "{m}"),
            other => panic!("unexpected {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn host_tool_implements_tool() {
        let host = Arc::new(MockMcp::default());
        let toolset = HostToolset::from_host(host).unwrap();
        let tool = toolset.get("json").unwrap();
        assert_eq!(tool.name(), "json");
        assert_eq!(Tool::description(tool), "the json tool");
        assert_eq!(tool.input_schema(), json!({"type": "object"}));
        let out = tool.run(HostToolInput(args())).unwrap();
        assert_eq!(out.result, json!({"n": 1}));
    }

    #[test]
    fn tool_spec_from_tool_info() {
        let info = ToolInfo {
            name: "a".into(),
            description: "b".into(),
            input_schema: "{}".into(),
        };
        let spec = crate::guest::ToolSpec::from(info.clone());
        assert_eq!(
            (spec.name, spec.description, spec.input_schema),
            (info.name, info.description, info.input_schema)
        );
    }
}
