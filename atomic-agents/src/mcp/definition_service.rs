use serde::{Deserialize, Serialize};

/// MCP transport protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpTransportType {
    /// Server-Sent Events (legacy).
    Sse,
    /// Streamable HTTP (recommended).
    HttpStream,
    /// Standard I/O (command-line process).
    Stdio,
}

/// Definition of an MCP tool, fetched from an MCP server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    pub output_schema: Option<serde_json::Value>,
}

/// Definition of an MCP resource.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpResourceDefinition {
    pub name: String,
    pub description: Option<String>,
    pub uri: String,
    pub input_schema: serde_json::Value,
    pub mime_type: Option<String>,
}

/// Definition of an MCP prompt template.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpPromptDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Service for fetching tool/resource/prompt definitions from MCP endpoints.
///
/// Connects to an MCP server via the configured transport and retrieves
/// available definitions. Mirrors Python's `MCPDefinitionService`.
pub struct McpDefinitionService {
    endpoint: String,
    transport_type: McpTransportType,
    working_directory: Option<String>,
}

impl McpDefinitionService {
    /// Creates a new MCP definition service.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - URL (for HTTP/SSE) or command string (for STDIO)
    /// * `transport_type` - Transport protocol to use
    pub fn new(endpoint: impl Into<String>, transport_type: McpTransportType) -> Self {
        Self {
            endpoint: endpoint.into(),
            transport_type,
            working_directory: None,
        }
    }

    /// Sets the working directory for STDIO commands.
    pub fn with_working_directory(mut self, dir: impl Into<String>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    /// Fetches tool definitions from the MCP server.
    ///
    /// Connects using the configured transport, calls `tools/list`,
    /// and returns the tool definitions with their input/output schemas.
    pub async fn fetch_tool_definitions(
        &self,
    ) -> Result<Vec<McpToolDefinition>, crate::error::Error> {
        match self.transport_type {
            McpTransportType::HttpStream | McpTransportType::Sse => {
                self.fetch_tools_http().await
            }
            McpTransportType::Stdio => self.fetch_tools_stdio().await,
        }
    }

    /// Fetches resource definitions from the MCP server.
    pub async fn fetch_resource_definitions(
        &self,
    ) -> Result<Vec<McpResourceDefinition>, crate::error::Error> {
        match self.transport_type {
            McpTransportType::HttpStream | McpTransportType::Sse => {
                self.fetch_resources_http().await
            }
            McpTransportType::Stdio => self.fetch_resources_stdio().await,
        }
    }

    /// Fetches prompt definitions from the MCP server.
    pub async fn fetch_prompt_definitions(
        &self,
    ) -> Result<Vec<McpPromptDefinition>, crate::error::Error> {
        match self.transport_type {
            McpTransportType::HttpStream | McpTransportType::Sse => {
                self.fetch_prompts_http().await
            }
            McpTransportType::Stdio => self.fetch_prompts_stdio().await,
        }
    }

    // --- HTTP/SSE transport ---

    async fn fetch_tools_http(&self) -> Result<Vec<McpToolDefinition>, crate::error::Error> {
        let client = reqwest::Client::new();
        let url = format!("{}/mcp/tools/list", self.endpoint.trim_end_matches('/'));

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({"jsonrpc": "2.0", "method": "tools/list", "id": 1}))
            .send()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("HTTP request failed: {e}")))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to parse response: {e}")))?;

        parse_tool_definitions(&body)
    }

    async fn fetch_resources_http(
        &self,
    ) -> Result<Vec<McpResourceDefinition>, crate::error::Error> {
        let client = reqwest::Client::new();
        let url = format!(
            "{}/mcp/resources/list",
            self.endpoint.trim_end_matches('/')
        );

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({"jsonrpc": "2.0", "method": "resources/list", "id": 1}))
            .send()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("HTTP request failed: {e}")))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to parse response: {e}")))?;

        parse_resource_definitions(&body)
    }

    async fn fetch_prompts_http(
        &self,
    ) -> Result<Vec<McpPromptDefinition>, crate::error::Error> {
        let client = reqwest::Client::new();
        let url = format!("{}/mcp/prompts/list", self.endpoint.trim_end_matches('/'));

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({"jsonrpc": "2.0", "method": "prompts/list", "id": 1}))
            .send()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("HTTP request failed: {e}")))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to parse response: {e}")))?;

        parse_prompt_definitions(&body)
    }

    // --- STDIO transport ---

    async fn fetch_tools_stdio(&self) -> Result<Vec<McpToolDefinition>, crate::error::Error> {
        let output = self.run_stdio_command("tools/list").await?;
        parse_tool_definitions(&output)
    }

    async fn fetch_resources_stdio(
        &self,
    ) -> Result<Vec<McpResourceDefinition>, crate::error::Error> {
        let output = self.run_stdio_command("resources/list").await?;
        parse_resource_definitions(&output)
    }

    async fn fetch_prompts_stdio(
        &self,
    ) -> Result<Vec<McpPromptDefinition>, crate::error::Error> {
        let output = self.run_stdio_command("prompts/list").await?;
        parse_prompt_definitions(&output)
    }

    async fn run_stdio_command(
        &self,
        method: &str,
    ) -> Result<serde_json::Value, crate::error::Error> {
        use tokio::process::Command;

        let parts: Vec<&str> = self.endpoint.split_whitespace().collect();
        if parts.is_empty() {
            return Err(crate::error::Error::Mcp(
                "STDIO endpoint command is empty".to_string(),
            ));
        }

        let mut cmd = Command::new(parts[0]);
        if parts.len() > 1 {
            cmd.args(&parts[1..]);
        }

        if let Some(dir) = &self.working_directory {
            cmd.current_dir(dir);
        }

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "id": 1
        });

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());

        let mut child = cmd
            .spawn()
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to spawn process: {e}")))?;

        if let Some(stdin) = child.stdin.as_mut() {
            use tokio::io::AsyncWriteExt;
            let request_str = serde_json::to_string(&request).unwrap();
            stdin
                .write_all(request_str.as_bytes())
                .await
                .map_err(|e| crate::error::Error::Mcp(format!("Failed to write to stdin: {e}")))?;
            stdin.shutdown().await.ok();
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("Process failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        serde_json::from_str(&stdout)
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to parse STDIO output: {e}")))
    }
}

fn parse_tool_definitions(
    body: &serde_json::Value,
) -> Result<Vec<McpToolDefinition>, crate::error::Error> {
    let tools = body
        .get("result")
        .and_then(|r| r.get("tools"))
        .and_then(|t| t.as_array())
        .ok_or_else(|| crate::error::Error::Mcp("No tools in response".to_string()))?;

    let mut definitions = Vec::new();
    for tool in tools {
        definitions.push(McpToolDefinition {
            name: tool
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            description: tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(String::from),
            input_schema: tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(serde_json::json!({})),
            output_schema: tool.get("outputSchema").cloned(),
        });
    }
    Ok(definitions)
}

fn parse_resource_definitions(
    body: &serde_json::Value,
) -> Result<Vec<McpResourceDefinition>, crate::error::Error> {
    let resources = body
        .get("result")
        .and_then(|r| r.get("resources"))
        .and_then(|t| t.as_array())
        .ok_or_else(|| crate::error::Error::Mcp("No resources in response".to_string()))?;

    let mut definitions = Vec::new();
    for res in resources {
        definitions.push(McpResourceDefinition {
            name: res
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            description: res
                .get("description")
                .and_then(|d| d.as_str())
                .map(String::from),
            uri: res
                .get("uri")
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string(),
            input_schema: res
                .get("inputSchema")
                .cloned()
                .unwrap_or(serde_json::json!({})),
            mime_type: res
                .get("mimeType")
                .and_then(|m| m.as_str())
                .map(String::from),
        });
    }
    Ok(definitions)
}

fn parse_prompt_definitions(
    body: &serde_json::Value,
) -> Result<Vec<McpPromptDefinition>, crate::error::Error> {
    let prompts = body
        .get("result")
        .and_then(|r| r.get("prompts"))
        .and_then(|t| t.as_array())
        .ok_or_else(|| crate::error::Error::Mcp("No prompts in response".to_string()))?;

    let mut definitions = Vec::new();
    for prompt in prompts {
        definitions.push(McpPromptDefinition {
            name: prompt
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            description: prompt
                .get("description")
                .and_then(|d| d.as_str())
                .map(String::from),
            input_schema: prompt
                .get("inputSchema")
                .cloned()
                .unwrap_or(serde_json::json!({})),
        });
    }
    Ok(definitions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_definitions() {
        let body = serde_json::json!({
            "result": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        });

        let defs = parse_tool_definitions(&body).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "search");
        assert_eq!(defs[0].description.as_deref(), Some("Search the web"));
    }

    #[test]
    fn test_parse_resource_definitions() {
        let body = serde_json::json!({
            "result": {
                "resources": [
                    {
                        "name": "config",
                        "description": "App configuration",
                        "uri": "config://app",
                        "mimeType": "application/json",
                        "inputSchema": {}
                    }
                ]
            }
        });

        let defs = parse_resource_definitions(&body).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "config");
        assert_eq!(defs[0].uri, "config://app");
    }

    #[test]
    fn test_parse_prompt_definitions() {
        let body = serde_json::json!({
            "result": {
                "prompts": [
                    {
                        "name": "summarize",
                        "description": "Summarize text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        });

        let defs = parse_prompt_definitions(&body).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "summarize");
    }
}
