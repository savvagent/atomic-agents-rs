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
mod definitions;
mod resource;
mod result;
mod tool;
mod transport;

pub use client::{McpClient, McpPaginationError};
pub use definitions::{
    McpPromptArgument, McpPromptDefinition, McpResourceDefinition, McpToolDefinition,
};
pub use resource::{McpResource, McpResourceContent, McpResourceOutput};
pub use rmcp::model::ServerCapabilities;
pub use tool::{McpTool, McpToolInput, McpToolOutput};
pub use transport::{McpTransport, McpTransportType};
