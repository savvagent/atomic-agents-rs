//! MCP (Model Context Protocol) client integration, built on [`rmcp`].
//!
//! An [`McpClient`] is one initialised session with an MCP server, reached
//! over stdio (a child process), Streamable HTTP or legacy HTTP+SSE. Discovery is gated by the
//! server's advertised capabilities and follows pagination; tool calls reuse
//! the session. Since Rust cannot create types at runtime, schemas and tool
//! results are `serde_json::Value`.
//!
//! - [`McpTransport`] / [`McpTransportType`] - how to reach the server
//! - [`McpClient`] - the session: discovery and `tools/call`
//! - [`McpTool`] - a discovered tool implementing [`Tool`](crate::tool::Tool)
//! - [`McpToolset`] / [`orchestrator_schema`] - one output schema over many tools, plus dispatch
//! - [`McpResource`] - a discovered resource or template implementing [`Resource`](crate::resource::Resource)
//! - [`McpPrompt`] - a discovered prompt implementing [`Prompt`](crate::prompt::Prompt)
//! - [`McpDefinitionService`] - one-shot discovery that opens and closes its own session
//! - [`McpToolDefinition`], [`McpResourceDefinition`], [`McpPromptDefinition`] - discovered metadata
//!
//! # Trust boundary
//!
//! A connected MCP server is inside your trust boundary. Its tool names,
//! descriptions and input schemas (and resource and prompt metadata) are
//! passed to the LLM as-is, unmarked, so a malicious server can steer the
//! model through them. [`McpToolset::dispatch`] only calls tools that were
//! discovered from the server, but arguments are not validated against a
//! tool's `input_schema` before they are sent. Connect only to servers you
//! trust.
//!
//! # Limits and isolation
//!
//! Each request is bounded by the transport's request timeout and connecting
//! by its connect timeout ([`McpTransport::with_request_timeout`],
//! [`McpTransport::with_connect_timeout`]); list requests stop after 1000
//! pages. A stdio server receives only [`DEFAULT_INHERITED_ENV_VARS`] from the
//! parent environment plus [`McpTransport::with_env`] values, unless
//! [`McpTransport::with_inherited_env`] is set.
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
mod redacting_http;
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
pub use transport::{
    DEFAULT_CONNECT_TIMEOUT, DEFAULT_INHERITED_ENV_VARS, DEFAULT_REQUEST_TIMEOUT, McpTransport,
    McpTransportType,
};
