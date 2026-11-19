//! MCP (Model Context Protocol) client integration, built on [`rmcp`].
//!
//! An [`McpClient`] is one initialised session with an MCP server, reached
//! over stdio (a child process) or Streamable HTTP. Discovery is gated by the
//! server's advertised capabilities and follows pagination; tool calls reuse
//! the session. Since Rust cannot create types at runtime, schemas and tool
//! results are `serde_json::Value`.
//!
//! - [`McpTransport`] / [`McpTransportType`] - how to reach the server
//! - [`McpClient`] - the session: discovery and `tools/call`
//! - [`McpTool`] - a discovered tool implementing [`Tool`](crate::tool::Tool)
//! - [`McpToolset`] / [`orchestrator_schema`] - one output schema over many tools, plus dispatch
//! - [`McpResource`] - a discovered resource or template implementing [`Resource`](crate::resource::Resource)
//! - [`McpToolDefinition`], [`McpResourceDefinition`], [`McpPromptDefinition`] - discovered metadata
//!
//! ```no_run
//! use atomic_agents::mcp::{McpClient, McpTransportType};
//!
//! # async fn demo() -> atomic_agents::error::Result<()> {
//! let client = McpClient::connect_endpoint("http://localhost:8080", McpTransportType::HttpStream).await?;
//! for tool in client.list_tools().await? {
//!     println!("{}", tool.name);
//! }
//! client.close().await?;
//! # Ok(())
//! # }
//! ```

mod client;
mod definition_service;
mod definitions;
mod legacy_sse;
mod orchestrator;
mod prompt;
mod resource;
mod result;
mod tool;
mod transport;

pub use client::{McpClient, McpPaginationError};
pub use definition_service::{McpDefinitionService, create_mcp_tools};
pub use definitions::{
    McpPromptArgument, McpPromptDefinition, McpResourceDefinition, McpToolDefinition,
};
pub use orchestrator::{
    McpOrchestratorOutput, McpToolCall, McpToolset, create_mcp_orchestrator_schema,
    fetch_mcp_attributes_with_schema, fetch_mcp_prompts, fetch_mcp_resources, fetch_mcp_tools,
    orchestrator_schema,
};
pub use prompt::{McpPrompt, McpPromptMessage, McpPromptOutput};
pub use resource::{McpResource, McpResourceContent, McpResourceOutput};
pub use rmcp::model::ServerCapabilities;
pub use tool::{McpTool, McpToolInput, McpToolOutput};
pub use transport::{McpTransport, McpTransportType};
