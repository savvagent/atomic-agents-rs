use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::pipeline::*;

/// Result of validating a pipeline definition.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationError {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationWarning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub message: String,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Validates a pipeline definition for structural and semantic correctness.
pub fn validate_pipeline(pipeline: &PipelineDefinition) -> ValidationReport {
    let mut report = ValidationReport::default();

    validate_node_ids_unique(pipeline, &mut report);
    validate_has_input_and_output(pipeline, &mut report);
    validate_edges_reference_valid_nodes(pipeline, &mut report);
    validate_schema_refs_exist(pipeline, &mut report);
    validate_no_cycles(pipeline, &mut report);
    validate_reachability(pipeline, &mut report);
    validate_router_conditions(pipeline, &mut report);
    validate_merge_inputs(pipeline, &mut report);

    report
}

fn validate_node_ids_unique(pipeline: &PipelineDefinition, report: &mut ValidationReport) {
    let mut seen = HashSet::new();
    for node in &pipeline.nodes {
        if !seen.insert(&node.id) {
            report.errors.push(ValidationError {
                node_id: Some(node.id.clone()),
                message: format!("Duplicate node ID: '{}'", node.id),
            });
        }
    }
}

fn validate_has_input_and_output(pipeline: &PipelineDefinition, report: &mut ValidationReport) {
    let has_input = pipeline
        .nodes
        .iter()
        .any(|n| matches!(n.config, NodeConfig::UserInput(_)));
    let has_output = pipeline
        .nodes
        .iter()
        .any(|n| matches!(n.config, NodeConfig::Output(_)));

    if !has_input {
        report.errors.push(ValidationError {
            node_id: None,
            message: "Pipeline must have at least one UserInput node".to_string(),
        });
    }
    if !has_output {
        report.errors.push(ValidationError {
            node_id: None,
            message: "Pipeline must have at least one Output node".to_string(),
        });
    }
}

fn validate_edges_reference_valid_nodes(
    pipeline: &PipelineDefinition,
    report: &mut ValidationReport,
) {
    let node_ids: HashSet<&str> = pipeline.nodes.iter().map(|n| n.id.as_str()).collect();
    for edge in &pipeline.edges {
        if !node_ids.contains(edge.source.as_str()) {
            report.errors.push(ValidationError {
                node_id: None,
                message: format!("Edge references unknown source node: '{}'", edge.source),
            });
        }
        if !node_ids.contains(edge.target.as_str()) {
            report.errors.push(ValidationError {
                node_id: None,
                message: format!("Edge references unknown target node: '{}'", edge.target),
            });
        }
    }
}

fn validate_schema_refs_exist(pipeline: &PipelineDefinition, report: &mut ValidationReport) {
    for node in &pipeline.nodes {
        let refs = match &node.config {
            NodeConfig::UserInput(c) => vec![c.schema_ref.as_str()],
            NodeConfig::Agent(c) => vec![c.input_schema.as_str(), c.output_schema.as_str()],
            NodeConfig::Merge(c) => c.output_schema.as_deref().into_iter().collect(),
            NodeConfig::Output(c) => c.schema_ref.as_deref().into_iter().collect(),
            _ => vec![],
        };

        for schema_ref in refs {
            if !pipeline.schemas.contains_key(schema_ref) {
                report.errors.push(ValidationError {
                    node_id: Some(node.id.clone()),
                    message: format!(
                        "Schema '{}' referenced by node '{}' not found in schemas",
                        schema_ref, node.id
                    ),
                });
            }
        }
    }
}

fn validate_no_cycles(pipeline: &PipelineDefinition, report: &mut ValidationReport) {
    // Kahn's algorithm to detect cycles
    let node_ids: Vec<&str> = pipeline.nodes.iter().map(|n| n.id.as_str()).collect();
    let mut in_degree: HashMap<&str, usize> = node_ids.iter().map(|id| (*id, 0)).collect();
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

    for edge in &pipeline.edges {
        *in_degree.entry(edge.target.as_str()).or_default() += 1;
        adj.entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }

    let mut queue: Vec<&str> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut visited = 0;
    while let Some(node) = queue.pop() {
        visited += 1;
        if let Some(neighbors) = adj.get(node) {
            for &next in neighbors {
                if let Some(d) = in_degree.get_mut(next) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push(next);
                    }
                }
            }
        }
    }

    if visited != node_ids.len() {
        report.errors.push(ValidationError {
            node_id: None,
            message: "Pipeline contains a cycle".to_string(),
        });
    }
}

fn validate_reachability(pipeline: &PipelineDefinition, report: &mut ValidationReport) {
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &pipeline.edges {
        adj.entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }

    // BFS from all input nodes
    let mut reachable = HashSet::new();
    let mut queue: Vec<&str> = pipeline
        .nodes
        .iter()
        .filter(|n| matches!(n.config, NodeConfig::UserInput(_)))
        .map(|n| n.id.as_str())
        .collect();

    while let Some(node) = queue.pop() {
        if reachable.insert(node) {
            if let Some(neighbors) = adj.get(node) {
                queue.extend(neighbors.iter());
            }
        }
    }

    for node in &pipeline.nodes {
        if !matches!(node.config, NodeConfig::UserInput(_)) && !reachable.contains(node.id.as_str())
        {
            report.warnings.push(ValidationWarning {
                node_id: Some(node.id.clone()),
                message: format!("Node '{}' is unreachable from any input", node.id),
            });
        }
    }
}

fn validate_router_conditions(pipeline: &PipelineDefinition, report: &mut ValidationReport) {
    for node in &pipeline.nodes {
        if let NodeConfig::Router(config) = &node.config {
            if config.conditions.is_empty() {
                report.errors.push(ValidationError {
                    node_id: Some(node.id.clone()),
                    message: "Router must have at least one condition".to_string(),
                });
            }

            // Check that condition names match edge handles
            let outgoing_handles: HashSet<&str> = pipeline
                .edges
                .iter()
                .filter(|e| e.source == node.id)
                .filter_map(|e| e.handle.as_deref())
                .collect();

            for cond in &config.conditions {
                if !outgoing_handles.contains(cond.name.as_str()) {
                    report.warnings.push(ValidationWarning {
                        node_id: Some(node.id.clone()),
                        message: format!(
                            "Router condition '{}' has no matching edge handle",
                            cond.name
                        ),
                    });
                }
            }
        }
    }
}

fn validate_merge_inputs(pipeline: &PipelineDefinition, report: &mut ValidationReport) {
    for node in &pipeline.nodes {
        if matches!(node.config, NodeConfig::Merge(_)) {
            let input_count = pipeline
                .edges
                .iter()
                .filter(|e| e.target == node.id)
                .count();
            if input_count < 2 {
                report.warnings.push(ValidationWarning {
                    node_id: Some(node.id.clone()),
                    message: format!(
                        "Merge node '{}' has {} inputs (expected 2+)",
                        node.id, input_count
                    ),
                });
            }
        }
    }
}

/// Checks if two JSON Schemas are structurally compatible.
///
/// Returns Ok(()) if all required fields in `target_schema` exist in
/// `source_schema` with compatible types.
pub fn check_schema_compatibility(
    source_schema: &serde_json::Value,
    target_schema: &serde_json::Value,
) -> Result<()> {
    let target_required: HashSet<String> = target_schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let source_props = source_schema.get("properties").and_then(|p| p.as_object());
    let target_props = target_schema.get("properties").and_then(|p| p.as_object());

    let empty_map = serde_json::Map::new();
    if let Some(target_props) = target_props {
        let source_props = source_props.unwrap_or(&empty_map);

        for field_name in &target_required {
            if !source_props.contains_key(field_name) {
                return Err(Error::SchemaValidation(format!(
                    "Required field '{}' in target schema not present in source schema",
                    field_name
                )));
            }
        }

        // Check type compatibility for shared fields
        for (field_name, target_field) in target_props {
            if let Some(source_field) = source_props.get(field_name) {
                let source_type = source_field.get("type").and_then(|t| t.as_str());
                let target_type = target_field.get("type").and_then(|t| t.as_str());

                if let (Some(st), Some(tt)) = (source_type, target_type) {
                    if st != tt {
                        return Err(Error::SchemaValidation(format!(
                            "Field '{}' type mismatch: source is '{}', target expects '{}'",
                            field_name, st, tt
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn simple_pipeline() -> PipelineDefinition {
        PipelineDefinition::from_json(
            r#"{
                "schemas": {
                    "Input": {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                    "Output": {"type": "object", "properties": {"reply": {"type": "string"}}}
                },
                "nodes": [
                    {"id": "in", "type": "user_input", "config": {"schema_ref": "Input"}},
                    {"id": "agent", "type": "agent", "config": {"input_schema": "Input", "output_schema": "Output"}},
                    {"id": "out", "type": "output", "config": {}}
                ],
                "edges": [
                    {"source": "in", "target": "agent"},
                    {"source": "agent", "target": "out"}
                ]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_valid_pipeline() {
        let report = validate_pipeline(&simple_pipeline());
        assert!(report.is_valid(), "Errors: {:?}", report.errors);
    }

    #[test]
    fn test_duplicate_node_ids() {
        let mut pipeline = simple_pipeline();
        pipeline.nodes.push(NodeDef {
            id: "in".to_string(),
            config: NodeConfig::UserInput(UserInputConfig {
                schema_ref: "Input".to_string(),
            }),
            position: [0.0, 0.0],
        });
        let report = validate_pipeline(&pipeline);
        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.message.contains("Duplicate"))
        );
    }

    #[test]
    fn test_missing_schema_ref() {
        let mut pipeline = simple_pipeline();
        if let NodeConfig::Agent(ref mut config) = pipeline.nodes[1].config {
            config.input_schema = "NonExistent".to_string();
        }
        let report = validate_pipeline(&pipeline);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_cycle_detection() {
        let pipeline = PipelineDefinition::from_json(
            r#"{
                "schemas": {"S": {"type": "object"}},
                "nodes": [
                    {"id": "in", "type": "user_input", "config": {"schema_ref": "S"}},
                    {"id": "a", "type": "agent", "config": {"input_schema": "S", "output_schema": "S"}},
                    {"id": "b", "type": "agent", "config": {"input_schema": "S", "output_schema": "S"}},
                    {"id": "out", "type": "output", "config": {}}
                ],
                "edges": [
                    {"source": "in", "target": "a"},
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "a"},
                    {"source": "b", "target": "out"}
                ]
            }"#,
        )
        .unwrap();
        let report = validate_pipeline(&pipeline);
        assert!(!report.is_valid());
        assert!(report.errors.iter().any(|e| e.message.contains("cycle")));
    }

    #[test]
    fn test_schema_compatibility() {
        let source = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });
        let target = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        assert!(check_schema_compatibility(&source, &target).is_ok());
    }

    #[test]
    fn test_schema_compatibility_missing_required() {
        let source = json!({
            "type": "object",
            "properties": {
                "age": {"type": "integer"}
            }
        });
        let target = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        assert!(check_schema_compatibility(&source, &target).is_err());
    }

    #[test]
    fn test_schema_compatibility_type_mismatch() {
        let source = json!({
            "type": "object",
            "properties": {
                "name": {"type": "integer"}
            }
        });
        let target = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        assert!(check_schema_compatibility(&source, &target).is_err());
    }
}
