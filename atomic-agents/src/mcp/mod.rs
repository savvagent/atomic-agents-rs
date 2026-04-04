//! MCP (Model Context Protocol) integration.
//!
//! Provides dynamic tool, resource, and prompt creation from MCP servers.
//! Since Rust cannot create types at runtime, MCP tools use `serde_json::Value`
//! for dynamic schemas with JSON Schema validation via the `jsonschema` crate.
//!
//! # Architecture
//!
//! - [`McpTransportType`] - Transport protocol selection (STDIO, HTTP, SSE)
//! - [`McpToolDefinition`] - Tool metadata fetched from MCP servers
//! - [`McpResourceDefinition`] - Resource metadata
//! - [`McpPromptDefinition`] - Prompt metadata
//! - [`McpDefinitionService`] - Fetches definitions from MCP endpoints
//! - [`McpTool`] - Dynamic tool implementation using `serde_json::Value` I/O

mod definition_service;
mod dynamic_tool;

pub use definition_service::{
    McpDefinitionService, McpPromptDefinition, McpResourceDefinition, McpToolDefinition,
    McpTransportType,
};
pub use dynamic_tool::McpTool;
