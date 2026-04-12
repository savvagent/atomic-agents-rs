use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Pipeline definition error: {0}")]
    Pipeline(String),

    #[error("Schema validation error: {0}")]
    SchemaValidation(String),

    #[error("Condition evaluation error: {0}")]
    Condition(String),

    #[error("Node execution error (node {node_id}): {message}")]
    NodeExecution { node_id: String, message: String },

    #[error("Graph error: {0}")]
    Graph(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Agent error: {0}")]
    Agent(#[from] atomic_agents::error::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
