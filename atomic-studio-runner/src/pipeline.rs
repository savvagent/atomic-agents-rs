use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A complete pipeline definition, serializable to/from JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineDefinition {
    /// Reusable JSON Schema definitions keyed by name.
    pub schemas: HashMap<String, serde_json::Value>,
    /// Node definitions.
    pub nodes: Vec<NodeDef>,
    /// Edge definitions connecting nodes.
    pub edges: Vec<EdgeDef>,
    /// Pipeline-level settings.
    #[serde(default)]
    pub settings: PipelineSettings,
}

/// A node in the pipeline graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeDef {
    /// Unique node identifier.
    pub id: String,
    /// Node type and configuration.
    #[serde(flatten)]
    pub config: NodeConfig,
    /// Canvas position [x, y] for the GUI (ignored by interpreter).
    #[serde(default)]
    pub position: [f64; 2],
}

/// Tagged union of node types and their configurations.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "snake_case")]
pub enum NodeConfig {
    /// Entry point: user provides input matching a schema.
    UserInput(UserInputConfig),
    /// An LLM agent that transforms input to output.
    Agent(AgentNodeConfig),
    /// A tool (builtin or MCP) that processes data.
    Tool(ToolNodeConfig),
    /// Conditional router that directs data to different paths.
    Router(RouterConfig),
    /// Merges multiple input paths into one output.
    Merge(MergeConfig),
    /// Terminal node that collects pipeline output.
    Output(OutputConfig),
}

/// Configuration for a UserInput node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserInputConfig {
    /// Key into `PipelineDefinition.schemas`.
    pub schema_ref: String,
}

/// Configuration for an Agent node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentNodeConfig {
    /// LLM provider ("openai", "anthropic", "google").
    #[serde(default)]
    pub provider: Option<String>,
    /// Model identifier.
    #[serde(default)]
    pub model: Option<String>,
    /// Key into `PipelineDefinition.schemas` for input.
    pub input_schema: String,
    /// Key into `PipelineDefinition.schemas` for output.
    pub output_schema: String,
    /// Extraction mode: "tools", "json", or "md_json".
    #[serde(default = "default_extraction_mode")]
    pub extraction_mode: String,
    /// System prompt configuration.
    #[serde(default)]
    pub system_prompt: SystemPromptConfig,
    /// Maximum retry attempts for structured output extraction.
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    /// Additional provider-specific parameters.
    #[serde(default)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// System prompt configuration for an agent node.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SystemPromptConfig {
    #[serde(default)]
    pub background: Vec<String>,
    #[serde(default)]
    pub steps: Vec<String>,
    #[serde(default)]
    pub output_instructions: Vec<String>,
}

/// Configuration for a Tool node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolNodeConfig {
    /// Tool source type.
    #[serde(flatten)]
    pub source: ToolSource,
}

/// Where a tool comes from.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "tool_type")]
#[serde(rename_all = "snake_case")]
pub enum ToolSource {
    /// A builtin tool from atomic-forge, referenced by name.
    Builtin { name: String },
    /// An MCP tool discovered from a server.
    Mcp {
        name: String,
        endpoint: String,
        #[serde(default = "default_transport")]
        transport: String,
    },
}

/// Configuration for a Router node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Ordered list of route conditions. First match wins.
    pub conditions: Vec<RouteCondition>,
    /// Name of the default route if no conditions match.
    pub default_route: String,
}

/// A single route condition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteCondition {
    /// Name of this route (matches edge handles).
    pub name: String,
    /// The condition expression to evaluate.
    #[serde(flatten)]
    pub expr: ConditionExpr,
}

/// A condition expression that can be evaluated against a JSON value.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op")]
#[serde(rename_all = "snake_case")]
pub enum ConditionExpr {
    /// Field value equals a string.
    Equals { field: String, value: serde_json::Value },
    /// Field value contains a substring.
    Contains { field: String, value: String },
    /// Field value is greater than a number.
    GreaterThan { field: String, value: f64 },
    /// Field value is less than a number.
    LessThan { field: String, value: f64 },
    /// Field value matches a regex pattern.
    Matches { field: String, pattern: String },
    /// Field exists and is not null.
    Exists { field: String },
    /// All sub-conditions must be true.
    And { conditions: Vec<ConditionExpr> },
    /// At least one sub-condition must be true.
    Or { conditions: Vec<ConditionExpr> },
    /// Negates a sub-condition.
    Not { condition: Box<ConditionExpr> },
}

/// Configuration for a Merge node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MergeConfig {
    /// How to combine inputs from multiple paths.
    pub strategy: MergeStrategy,
    /// Key into `PipelineDefinition.schemas` for the merged output.
    #[serde(default)]
    pub output_schema: Option<String>,
}

/// Strategy for merging multiple inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// Wait for all inputs, collect into a JSON array.
    WaitAll,
    /// Proceed on the first input that arrives.
    FirstReady,
    /// Deep-merge JSON objects from all inputs.
    MergeObjects,
}

/// Configuration for an Output node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Key into `PipelineDefinition.schemas`.
    #[serde(default)]
    pub schema_ref: Option<String>,
}

/// An edge connecting two nodes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeDef {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Optional handle name (for router condition outputs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handle: Option<String>,
}

/// Pipeline-level settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSettings {
    /// Default LLM provider for agent nodes that don't specify one.
    #[serde(default = "default_provider")]
    pub default_provider: String,
    /// Default model for agent nodes that don't specify one.
    #[serde(default = "default_model")]
    pub default_model: String,
    /// Maximum number of nodes to execute concurrently.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            default_provider: default_provider(),
            default_model: default_model(),
            max_concurrent: default_max_concurrent(),
        }
    }
}

fn default_extraction_mode() -> String {
    "tools".to_string()
}

fn default_max_retries() -> usize {
    3
}

fn default_transport() -> String {
    "http_stream".to_string()
}

fn default_provider() -> String {
    "openai".to_string()
}

fn default_model() -> String {
    "gpt-4o".to_string()
}

fn default_max_concurrent() -> usize {
    4
}

impl PipelineDefinition {
    /// Loads a pipeline definition from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serializes the pipeline definition to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Loads a pipeline definition from a file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, crate::error::Error> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::Error::Io(e.to_string()))?;
        Self::from_json(&content).map_err(|e| crate::error::Error::Pipeline(e.to_string()))
    }

    /// Saves the pipeline definition to a file.
    pub fn to_file(&self, path: &std::path::Path) -> Result<(), crate::error::Error> {
        let json = self.to_json().map_err(|e| crate::error::Error::Pipeline(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| crate::error::Error::Io(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple_pipeline() {
        let json = r#"{
            "schemas": {
                "Input": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                "Output": {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
            },
            "nodes": [
                {"id": "in", "type": "user_input", "config": {"schema_ref": "Input"}, "position": [0, 0]},
                {"id": "agent", "type": "agent", "config": {
                    "input_schema": "Input", "output_schema": "Output",
                    "system_prompt": {"background": ["You are helpful."]}
                }, "position": [300, 0]},
                {"id": "out", "type": "output", "config": {}, "position": [600, 0]}
            ],
            "edges": [
                {"source": "in", "target": "agent"},
                {"source": "agent", "target": "out"}
            ]
        }"#;

        let pipeline = PipelineDefinition::from_json(json).unwrap();
        assert_eq!(pipeline.nodes.len(), 3);
        assert_eq!(pipeline.edges.len(), 2);
        assert_eq!(pipeline.schemas.len(), 2);

        // Roundtrip
        let json2 = pipeline.to_json().unwrap();
        let pipeline2 = PipelineDefinition::from_json(&json2).unwrap();
        assert_eq!(pipeline2.nodes.len(), 3);
    }

    #[test]
    fn test_router_pipeline() {
        let json = r#"{
            "schemas": {
                "Ticket": {"type": "object", "properties": {"urgency": {"type": "string"}}},
                "Response": {"type": "object", "properties": {"reply": {"type": "string"}}}
            },
            "nodes": [
                {"id": "in", "type": "user_input", "config": {"schema_ref": "Ticket"}, "position": [0, 0]},
                {"id": "router", "type": "router", "config": {
                    "conditions": [
                        {"name": "urgent", "op": "equals", "field": "urgency", "value": "high"},
                        {"name": "normal", "op": "equals", "field": "urgency", "value": "low"}
                    ],
                    "default_route": "normal"
                }, "position": [300, 0]},
                {"id": "out", "type": "output", "config": {}, "position": [600, 0]}
            ],
            "edges": [
                {"source": "in", "target": "router"},
                {"source": "router", "target": "out", "handle": "urgent"},
                {"source": "router", "target": "out", "handle": "normal"}
            ]
        }"#;

        let pipeline = PipelineDefinition::from_json(json).unwrap();
        assert_eq!(pipeline.nodes.len(), 3);

        if let NodeConfig::Router(ref config) = pipeline.nodes[1].config {
            assert_eq!(config.conditions.len(), 2);
            assert_eq!(config.default_route, "normal");
        } else {
            panic!("Expected router node");
        }
    }

    #[test]
    fn test_default_settings() {
        let json = r#"{
            "schemas": {},
            "nodes": [],
            "edges": []
        }"#;

        let pipeline = PipelineDefinition::from_json(json).unwrap();
        assert_eq!(pipeline.settings.default_provider, "openai");
        assert_eq!(pipeline.settings.default_model, "gpt-4o");
        assert_eq!(pipeline.settings.max_concurrent, 4);
    }
}
