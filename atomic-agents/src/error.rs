use thiserror::Error;

/// Top-level error type for the atomic-agents framework.
#[derive(Error, Debug)]
pub enum Error {
    #[error("LLM client error: {0}")]
    Client(#[from] ClientError),

    #[error("Schema validation failed: {0}")]
    Validation(String),

    #[error("Structured output extraction failed after {attempts} attempts: {message}")]
    ExtractionFailed { attempts: usize, message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Chat history error: {0}")]
    History(String),

    #[error("Token counting error: {0}")]
    TokenCount(String),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Error type for LLM client operations.
#[derive(Error, Debug)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },

    #[error("Rate limited, retry after {retry_after:?}s")]
    RateLimited { retry_after: Option<u64> },

    #[error("Authentication failed")]
    AuthFailed,

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, Error>;
