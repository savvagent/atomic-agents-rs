//! Pipeline spec types (serde).
//!
//! Unknown fields are ignored so newer specs degrade gracefully on an older
//! runner; [`crate::validate::validate`] enforces the rules serde cannot.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::condition::ConditionExpr;

/// Upper bound on `foreach.max_items`.
pub const MAX_FOREACH_ITEMS: u32 = 100;
/// `max_turns` used by an agent step that sets none.
pub const DEFAULT_MAX_TURNS: u32 = 8;
/// Upper bound on `max_turns`.
pub const MAX_MAX_TURNS: u32 = 32;
/// `max_retries` used when neither the step nor the pipeline sets one.
pub const DEFAULT_MAX_RETRIES: u32 = 2;
/// Upper bound on `max_retries`.
pub const MAX_MAX_RETRIES: u32 = 5;
/// Tool calls one run may make.
pub const MAX_TOOL_CALLS: u32 = 500;
/// LLM calls one run may make.
pub const MAX_LLM_CALLS: u32 = 200;

/// A whole pipeline.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pipeline {
    /// Spec format version; only `1` is accepted.
    pub spec_version: u32,
    /// Human-readable name.
    pub name: String,
    /// Default model for agent steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Default retry budget for agent steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// Steps, run in order.
    pub steps: Vec<Step>,
}

/// One step: common fields plus a kind-specific body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Step {
    /// Identifier; names the step's output in the template context.
    pub id: String,
    /// Run the step only when this holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
    /// State keys to write after the step, each a template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set_state: Option<BTreeMap<String, Value>>,
    /// Kind-specific body.
    #[serde(flatten)]
    pub kind: StepKind,
    /// Keys no other field claimed. Kept so validation can reject a
    /// `constraints` key on a non-act step, which serde would otherwise
    /// drop silently; never serialized.
    #[serde(flatten, default, skip_serializing)]
    pub extras: Map<String, Value>,
}

/// The four step kinds, tagged by `kind`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StepKind {
    /// An LLM call with structured output and optional read tools.
    Agent(AgentStep),
    /// A single tool call.
    Tool(ToolStep),
    /// A tool call that changes something, guarded by constraints.
    Act(ActStep),
    /// Run nested steps once per item.
    Foreach(ForeachStep),
}

/// An LLM call with structured output.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentStep {
    /// System prompt (templated).
    pub system_prompt: String,
    /// Input value (templated).
    pub input: Value,
    /// JSON Schema the output must satisfy.
    pub output_schema: Value,
    /// Model; falls back to the pipeline's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Tool-use turns allowed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    /// Retries on invalid output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// Names of the tools the model may call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// Extra model parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Map<String, Value>>,
}

/// A single tool call.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolStep {
    /// Tool name.
    pub tool: String,
    /// Arguments (templated).
    pub args: Value,
}

/// A guarded tool call.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ActStep {
    /// Tool name.
    pub tool: String,
    /// Arguments (templated).
    pub args: Value,
    /// Per-argument constraints checked before the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraints: Option<BTreeMap<String, Constraint>>,
}

/// Limits on one argument of an `act` step. Every present field must hold.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Constraint {
    /// Value must be one of these.
    #[serde(default, rename = "in", skip_serializing_if = "Option::is_none")]
    pub in_: Option<Vec<Value>>,
    /// Value must equal this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eq: Option<Value>,
    /// Number must be below this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lt: Option<f64>,
    /// Number must be at most this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lte: Option<f64>,
    /// Number must be above this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gt: Option<f64>,
    /// Number must be at least this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gte: Option<f64>,
    /// String must be at most this many characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
}

/// Run nested steps once per item.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForeachStep {
    /// Path of the array to iterate.
    pub items: String,
    /// Context name of the current item; `item` when absent.
    #[serde(default, rename = "as", skip_serializing_if = "Option::is_none")]
    pub as_: Option<String>,
    /// Most items processed; more is a failure.
    pub max_items: u32,
    /// Steps run per item.
    pub steps: Vec<Step>,
}

impl ForeachStep {
    /// The context name of the current item.
    pub fn item_name(&self) -> &str {
        self.as_.as_deref().unwrap_or("item")
    }
}

/// A `when` guard: an expression string or the structured form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Condition {
    /// Expression such as `a.b == 'x' && c > 1`.
    Expr(String),
    /// Structured form shared with the studio runner.
    Structured(ConditionExpr),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> Value {
        json!({
            "spec_version": 1,
            "name": "demo",
            "model": "m1",
            "max_retries": 1,
            "future_field": true,
            "steps": [
                {"id": "a", "kind": "agent", "system_prompt": "p", "input": {"x": "{{ event.x }}"},
                 "output_schema": {"type": "object"}, "max_turns": 3, "tools": ["read"],
                 "params": {"temperature": 0.2}},
                {"id": "t", "kind": "tool", "tool": "read", "args": {"k": 1},
                 "when": "a.ok == true", "set_state": {"last": "{{ a }}"}},
                {"id": "c", "kind": "act", "tool": "write", "args": {"n": 2},
                 "when": {"op": "exists", "field": "a"},
                 "constraints": {"n": {"in": [1, 2], "lte": 5.0, "max_length": 3}}},
                {"id": "f", "kind": "foreach", "items": "a.list", "as": "it", "max_items": 10,
                 "steps": [{"id": "g", "kind": "tool", "tool": "read", "args": {}}]}
            ]
        })
    }

    #[test]
    fn round_trips_all_four_kinds() {
        let p: Pipeline = serde_json::from_value(sample()).unwrap();
        assert_eq!(p.steps.len(), 4);
        assert!(matches!(p.steps[0].kind, StepKind::Agent(_)));
        assert!(matches!(p.steps[1].kind, StepKind::Tool(_)));
        assert!(matches!(p.steps[2].kind, StepKind::Act(_)));
        assert!(matches!(&p.steps[3].kind, StepKind::Foreach(f) if f.item_name() == "it"));
        assert!(matches!(p.steps[1].when, Some(Condition::Expr(_))));
        assert!(matches!(p.steps[2].when, Some(Condition::Structured(_))));
        let again: Pipeline = serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
        assert_eq!(p, again);
    }

    #[test]
    fn constraint_fields_parse() {
        let p: Pipeline = serde_json::from_value(sample()).unwrap();
        let StepKind::Act(a) = &p.steps[2].kind else {
            panic!("not act")
        };
        let c = &a.constraints.as_ref().unwrap()["n"];
        assert_eq!(c.in_, Some(vec![json!(1), json!(2)]));
        assert_eq!(c.lte, Some(5.0));
        assert_eq!(c.max_length, Some(3));
    }

    #[test]
    fn constraints_parse_on_act_and_land_in_extras_elsewhere() {
        let p: Pipeline = serde_json::from_value(sample()).unwrap();
        let StepKind::Act(a) = &p.steps[2].kind else {
            panic!("not act")
        };
        assert!(a.constraints.is_some());
        let t: Step = serde_json::from_value(
            json!({"id": "t", "kind": "tool", "tool": "x", "args": {}, "constraints": {}}),
        )
        .unwrap();
        assert!(t.extras.contains_key("constraints"));
    }

    #[test]
    fn default_item_name_and_unknown_kind() {
        let f: ForeachStep =
            serde_json::from_value(json!({"items": "x", "max_items": 1, "steps": []})).unwrap();
        assert_eq!(f.item_name(), "item");
        let bad = json!({"spec_version": 1, "name": "n",
            "steps": [{"id": "a", "kind": "nope"}]});
        assert!(serde_json::from_value::<Pipeline>(bad).is_err());
    }
}
