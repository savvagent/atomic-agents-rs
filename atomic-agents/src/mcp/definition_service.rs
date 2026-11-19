//! One-shot discovery, kept for callers of the pre-session API.

use std::path::PathBuf;

use super::client::McpClient;
use super::definitions::{McpPromptDefinition, McpResourceDefinition, McpToolDefinition};
use super::orchestrator::fetch_mcp_tools;
use super::tool::McpTool;
use super::transport::{McpTransport, McpTransportType};
use crate::error::Result;

/// Fetches definitions from an MCP server, one short-lived session per call.
///
/// Each `fetch_*` method connects, lists, and closes the session. Prefer
/// [`McpClient`] when you need more than one request: it keeps one session
/// open and lets tools call back into it.
#[derive(Clone, Debug)]
pub struct McpDefinitionService {
    endpoint: String,
    transport_type: McpTransportType,
    working_directory: Option<PathBuf>,
}

impl McpDefinitionService {
    /// A service for `endpoint`, interpreted by [`McpTransport::from_endpoint`].
    pub fn new(endpoint: impl Into<String>, transport_type: McpTransportType) -> Self {
        Self {
            endpoint: endpoint.into(),
            transport_type,
            working_directory: None,
        }
    }

    /// Sets the working directory of a stdio server. Ignored for HTTP transports.
    pub fn with_working_directory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    /// Lists the server's tools.
    pub async fn fetch_tool_definitions(&self) -> Result<Vec<McpToolDefinition>> {
        self.with_session(|c| async move { c.list_tools().await })
            .await
    }

    /// Lists the server's resources and resource templates.
    pub async fn fetch_resource_definitions(&self) -> Result<Vec<McpResourceDefinition>> {
        self.with_session(|c| async move { c.list_resources().await })
            .await
    }

    /// Lists the server's prompts.
    pub async fn fetch_prompt_definitions(&self) -> Result<Vec<McpPromptDefinition>> {
        self.with_session(|c| async move { c.list_prompts().await })
            .await
    }

    /// Connects, runs `f`, and closes the session. A failure to close after
    /// a successful `f` is ignored: the answer is already in hand and the
    /// session is dropped either way.
    async fn with_session<T, F, Fut>(&self, f: F) -> Result<T>
    where
        F: FnOnce(McpClient) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut transport =
            McpTransport::from_endpoint(&self.endpoint, self.transport_type.clone())?;
        if let Some(dir) = &self.working_directory {
            transport = transport.with_cwd(dir.clone());
        }
        let client = McpClient::connect(transport).await?;
        let result = f(client.clone()).await;
        if let Err(e) = client.close().await {
            tracing::debug!("closing one-shot MCP session failed: {e}");
        }
        result
    }
}

/// Connects and lists the server's tools, like [`fetch_mcp_tools`].
///
/// The returned [`McpClient`] owns the session the tools call through.
pub async fn create_mcp_tools(
    endpoint: &str,
    kind: McpTransportType,
) -> Result<(McpClient, Vec<McpTool>)> {
    fetch_mcp_tools(endpoint, kind).await
}
