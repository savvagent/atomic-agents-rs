use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;
use tokio::sync::{Mutex, Semaphore, broadcast};

use crate::condition;
use crate::dynamic_agent::DynamicAgent;
use crate::dynamic_tool::DynamicToolRunner;
use crate::error::{Error, Result};
use crate::merge::merge_values;
use crate::pipeline::*;

/// Events emitted during pipeline execution.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum ExecutionEvent {
    NodeStarted { node_id: String },
    NodeCompleted { node_id: String, output: Value },
    NodeFailed { node_id: String, error: String },
    PipelineCompleted { outputs: HashMap<String, Value> },
    PipelineFailed { error: String },
}

/// Factory for creating LLM clients from provider names.
pub trait ClientFactory: Send + Sync {
    fn create_client(
        &self,
        provider: &str,
    ) -> std::result::Result<Box<dyn atomic_agents::client::LlmClient>, Error>;
}

/// Factory for creating dynamic tool runners from tool source configs.
pub trait ToolFactory: Send + Sync {
    fn create_tool(
        &self,
        source: &ToolSource,
    ) -> std::result::Result<Box<dyn DynamicToolRunner>, Error>;
}

/// Executes a pipeline definition as a DAG.
pub struct PipelineInterpreter {
    pipeline: PipelineDefinition,
    client_factory: Box<dyn ClientFactory>,
    tool_factory: Option<Box<dyn ToolFactory>>,
    event_tx: broadcast::Sender<ExecutionEvent>,
}

impl PipelineInterpreter {
    pub fn new(pipeline: PipelineDefinition, client_factory: Box<dyn ClientFactory>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            pipeline,
            client_factory,
            tool_factory: None,
            event_tx,
        }
    }

    pub fn with_tool_factory(mut self, factory: Box<dyn ToolFactory>) -> Self {
        self.tool_factory = Some(factory);
        self
    }

    /// Subscribe to execution events.
    pub fn subscribe(&self) -> broadcast::Receiver<ExecutionEvent> {
        self.event_tx.subscribe()
    }

    /// Executes the pipeline with the given initial input.
    ///
    /// Returns the outputs from all Output nodes.
    pub async fn run(&self, input: Value) -> Result<HashMap<String, Value>> {
        let node_map: HashMap<&str, &NodeDef> = self
            .pipeline
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), n))
            .collect();

        // Build adjacency: node_id -> Vec<(target_id, optional_handle)>
        let mut outgoing: HashMap<&str, Vec<(&str, Option<&str>)>> = HashMap::new();
        let mut incoming: HashMap<&str, Vec<(&str, Option<&str>)>> = HashMap::new();
        for edge in &self.pipeline.edges {
            outgoing
                .entry(edge.source.as_str())
                .or_default()
                .push((edge.target.as_str(), edge.handle.as_deref()));
            incoming
                .entry(edge.target.as_str())
                .or_default()
                .push((edge.source.as_str(), edge.handle.as_deref()));
        }

        // Node outputs storage
        let outputs: Arc<Mutex<HashMap<String, Value>>> = Arc::new(Mutex::new(HashMap::new()));

        // In-degree tracking for topological execution
        let in_degree: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));
        {
            let mut deg = in_degree.lock().await;
            for node in &self.pipeline.nodes {
                let count = incoming.get(node.id.as_str()).map_or(0, |v| v.len());
                deg.insert(node.id.clone(), count);
            }
        }

        // Merge node input collectors
        let merge_inputs: Arc<Mutex<HashMap<String, Vec<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        for node in &self.pipeline.nodes {
            if matches!(node.config, NodeConfig::Merge(_)) {
                merge_inputs
                    .lock()
                    .await
                    .insert(node.id.clone(), Vec::new());
            }
        }

        // Seed user input nodes
        {
            let mut out = outputs.lock().await;
            for node in &self.pipeline.nodes {
                if matches!(node.config, NodeConfig::UserInput(_)) {
                    out.insert(node.id.clone(), input.clone());
                }
            }
        }

        let semaphore = Arc::new(Semaphore::new(self.pipeline.settings.max_concurrent));
        let (ready_tx, ready_rx) = async_channel::unbounded::<String>();

        // Enqueue initially ready nodes (in-degree 0)
        {
            let deg = in_degree.lock().await;
            for (id, &d) in deg.iter() {
                if d == 0 {
                    ready_tx.send(id.clone()).await.ok();
                }
            }
        }

        let active_count = Arc::new(Mutex::new(0usize));

        loop {
            // Try to receive a ready node, with a timeout to check if we're done
            let node_id =
                match tokio::time::timeout(std::time::Duration::from_millis(100), ready_rx.recv())
                    .await
                {
                    Ok(Ok(id)) => id,
                    Ok(Err(_)) => break, // Channel closed
                    Err(_) => {
                        // Timeout — check if all done
                        let active = *active_count.lock().await;
                        if active == 0 && ready_rx.is_empty() {
                            break;
                        }
                        continue;
                    }
                };

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            *active_count.lock().await += 1;

            let node = match node_map.get(node_id.as_str()) {
                Some(n) => (*n).clone(),
                None => continue,
            };

            let outputs = outputs.clone();
            let in_degree = in_degree.clone();
            let merge_inputs = merge_inputs.clone();
            let ready_tx = ready_tx.clone();
            let event_tx = self.event_tx.clone();
            let active_count = active_count.clone();

            // Collect info needed for execution
            let incoming_edges: Vec<(String, Option<String>)> = incoming
                .get(node_id.as_str())
                .map(|v| {
                    v.iter()
                        .map(|(s, h)| (s.to_string(), h.map(|h| h.to_string())))
                        .collect()
                })
                .unwrap_or_default();

            let outgoing_edges: Vec<(String, Option<String>)> = outgoing
                .get(node_id.as_str())
                .map(|v| {
                    v.iter()
                        .map(|(t, h)| (t.to_string(), h.map(|h| h.to_string())))
                        .collect()
                })
                .unwrap_or_default();

            let schemas = self.pipeline.schemas.clone();
            let settings = self.pipeline.settings.clone();
            let client_factory_result = self.create_node_client(&node, &settings);
            let tool_runner = self.create_node_tool(&node);

            tokio::spawn(async move {
                event_tx
                    .send(ExecutionEvent::NodeStarted {
                        node_id: node_id.clone(),
                    })
                    .ok();

                let result = execute_node(
                    &node,
                    &incoming_edges,
                    &outputs,
                    &merge_inputs,
                    &schemas,
                    &settings,
                    client_factory_result,
                    tool_runner,
                )
                .await;

                match result {
                    Ok(output) => {
                        // Store output
                        outputs.lock().await.insert(node_id.clone(), output.clone());

                        event_tx
                            .send(ExecutionEvent::NodeCompleted {
                                node_id: node_id.clone(),
                                output: output.clone(),
                            })
                            .ok();

                        // Determine which downstream edges to activate
                        let activated_edges = match &node.config {
                            NodeConfig::Router(router_config) => {
                                // Evaluate conditions to determine which route
                                let route = evaluate_router(router_config, &output);
                                outgoing_edges
                                    .iter()
                                    .filter(|(_, handle)| handle.as_deref() == Some(route.as_str()))
                                    .cloned()
                                    .collect::<Vec<_>>()
                            }
                            _ => outgoing_edges.clone(),
                        };

                        // Decrement in-degree for activated targets
                        let mut deg = in_degree.lock().await;
                        for (target_id, _) in &activated_edges {
                            if let Some(d) = deg.get_mut(target_id.as_str()) {
                                *d = d.saturating_sub(1);

                                // Handle merge node input collection
                                {
                                    let mut mi = merge_inputs.lock().await;
                                    if let Some(inputs) = mi.get_mut(target_id.as_str()) {
                                        inputs.push(output.clone());
                                    }
                                }

                                if *d == 0 {
                                    ready_tx.send(target_id.clone()).await.ok();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        event_tx
                            .send(ExecutionEvent::NodeFailed {
                                node_id: node_id.clone(),
                                error: e.to_string(),
                            })
                            .ok();
                        event_tx
                            .send(ExecutionEvent::PipelineFailed {
                                error: format!("Node '{}' failed: {}", node_id, e),
                            })
                            .ok();
                    }
                }

                *active_count.lock().await -= 1;
                drop(permit);
            });
        }

        // Collect outputs from Output nodes
        let final_outputs = outputs.lock().await;
        let mut result = HashMap::new();
        for node in &self.pipeline.nodes {
            if matches!(node.config, NodeConfig::Output(_)) {
                if let Some(output) = final_outputs.get(&node.id) {
                    result.insert(node.id.clone(), output.clone());
                }
            }
        }

        self.event_tx
            .send(ExecutionEvent::PipelineCompleted {
                outputs: result.clone(),
            })
            .ok();

        Ok(result)
    }

    fn create_node_client(
        &self,
        node: &NodeDef,
        settings: &PipelineSettings,
    ) -> Option<std::result::Result<Box<dyn atomic_agents::client::LlmClient>, Error>> {
        if let NodeConfig::Agent(config) = &node.config {
            let provider = config
                .provider
                .as_deref()
                .unwrap_or(&settings.default_provider);
            Some(self.client_factory.create_client(provider))
        } else {
            None
        }
    }

    fn create_node_tool(
        &self,
        node: &NodeDef,
    ) -> Option<std::result::Result<Box<dyn DynamicToolRunner>, Error>> {
        if let NodeConfig::Tool(config) = &node.config {
            self.tool_factory
                .as_ref()
                .map(|f| f.create_tool(&config.source))
        } else {
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_node(
    node: &NodeDef,
    incoming_edges: &[(String, Option<String>)],
    outputs: &Arc<Mutex<HashMap<String, Value>>>,
    merge_inputs: &Arc<Mutex<HashMap<String, Vec<Value>>>>,
    schemas: &HashMap<String, Value>,
    settings: &PipelineSettings,
    client_result: Option<std::result::Result<Box<dyn atomic_agents::client::LlmClient>, Error>>,
    tool_result: Option<std::result::Result<Box<dyn DynamicToolRunner>, Error>>,
) -> Result<Value> {
    match &node.config {
        NodeConfig::UserInput(_) => {
            // Already seeded in outputs during initialization
            let out = outputs.lock().await;
            out.get(&node.id)
                .cloned()
                .ok_or_else(|| Error::NodeExecution {
                    node_id: node.id.clone(),
                    message: "No input provided".to_string(),
                })
        }

        NodeConfig::Agent(config) => {
            let input = get_upstream_output(incoming_edges, outputs).await?;

            let input_schema = schemas
                .get(&config.input_schema)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type": "object"}));
            let output_schema = schemas
                .get(&config.output_schema)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type": "object"}));

            let client = client_result
                .ok_or_else(|| Error::NodeExecution {
                    node_id: node.id.clone(),
                    message: "No client factory result".to_string(),
                })?
                .map_err(|e| Error::NodeExecution {
                    node_id: node.id.clone(),
                    message: format!("Failed to create client: {e}"),
                })?;

            let model = config
                .model
                .as_deref()
                .unwrap_or(&settings.default_model)
                .to_string();

            let mut agent = DynamicAgent::new(client, config, model, input_schema, output_schema);

            agent.run(input).await.map_err(|e| Error::NodeExecution {
                node_id: node.id.clone(),
                message: e.to_string(),
            })
        }

        NodeConfig::Tool(_config) => {
            let input = get_upstream_output(incoming_edges, outputs).await?;

            let runner = tool_result
                .ok_or_else(|| Error::NodeExecution {
                    node_id: node.id.clone(),
                    message: "No tool factory configured".to_string(),
                })?
                .map_err(|e| Error::NodeExecution {
                    node_id: node.id.clone(),
                    message: format!("Failed to create tool: {e}"),
                })?;

            runner.run(input).await.map_err(|e| Error::NodeExecution {
                node_id: node.id.clone(),
                message: e.to_string(),
            })
        }

        NodeConfig::Router(_) => {
            // Router passes through its input; routing logic is in the post-execution step
            get_upstream_output(incoming_edges, outputs).await
        }

        NodeConfig::Merge(config) => {
            let inputs = merge_inputs
                .lock()
                .await
                .get(&node.id)
                .cloned()
                .unwrap_or_default();

            merge_values(&config.strategy, inputs).map_err(|e| Error::NodeExecution {
                node_id: node.id.clone(),
                message: e.to_string(),
            })
        }

        NodeConfig::Output(_) => {
            // Pass through upstream output
            get_upstream_output(incoming_edges, outputs).await
        }
    }
}

/// Gets the output of the first upstream node.
async fn get_upstream_output(
    incoming_edges: &[(String, Option<String>)],
    outputs: &Arc<Mutex<HashMap<String, Value>>>,
) -> Result<Value> {
    let out = outputs.lock().await;
    for (source_id, _) in incoming_edges {
        if let Some(value) = out.get(source_id) {
            return Ok(value.clone());
        }
    }
    Err(Error::Pipeline("No upstream output available".to_string()))
}

/// Evaluates router conditions and returns the selected route name.
fn evaluate_router(config: &RouterConfig, data: &Value) -> String {
    for route in &config.conditions {
        match condition::evaluate(&route.expr, data) {
            Ok(true) => return route.name.clone(),
            _ => continue,
        }
    }
    config.default_route.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_evaluate_router() {
        let config = RouterConfig {
            conditions: vec![
                RouteCondition {
                    name: "urgent".to_string(),
                    expr: ConditionExpr::Equals {
                        field: "urgency".to_string(),
                        value: json!("high"),
                    },
                },
                RouteCondition {
                    name: "normal".to_string(),
                    expr: ConditionExpr::Equals {
                        field: "urgency".to_string(),
                        value: json!("low"),
                    },
                },
            ],
            default_route: "normal".to_string(),
        };

        assert_eq!(
            evaluate_router(&config, &json!({"urgency": "high"})),
            "urgent"
        );
        assert_eq!(
            evaluate_router(&config, &json!({"urgency": "low"})),
            "normal"
        );
        assert_eq!(
            evaluate_router(&config, &json!({"urgency": "medium"})),
            "normal"
        );
    }
}
