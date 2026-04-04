use std::sync::Arc;

use super::definition_service::{McpDefinitionService, McpToolDefinition, McpTransportType};

/// A dynamically-created tool from an MCP server.
///
/// Since Rust cannot create types at runtime, MCP tools use `serde_json::Value`
/// for both input and output. The tool stores its JSON Schema for validation
/// and connects to the MCP server to execute.
///
/// Mirrors Python's dynamically-generated `BaseTool` subclasses from `MCPFactory`.
#[derive(Clone)]
pub struct McpTool {
    /// The tool name from the MCP server.
    pub name: String,
    /// The tool description.
    pub description: Option<String>,
    /// JSON Schema for the input parameters.
    pub input_schema: serde_json::Value,
    /// JSON Schema for the output (if provided by the server).
    pub output_schema: Option<serde_json::Value>,
    /// MCP endpoint URL or command.
    endpoint: String,
    /// Transport type for connecting to the MCP server.
    transport_type: McpTransportType,
    /// Working directory for STDIO transport.
    working_directory: Option<String>,
}

impl McpTool {
    /// Creates a new MCP tool from a definition and connection details.
    pub fn new(
        definition: McpToolDefinition,
        endpoint: impl Into<String>,
        transport_type: McpTransportType,
    ) -> Self {
        Self {
            name: definition.name,
            description: definition.description,
            input_schema: definition.input_schema,
            output_schema: definition.output_schema,
            endpoint: endpoint.into(),
            transport_type,
            working_directory: None,
        }
    }

    /// Sets the working directory for STDIO transport.
    pub fn with_working_directory(mut self, dir: impl Into<String>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    /// Executes the tool with the given input parameters.
    ///
    /// Connects to the MCP server, calls the tool, and returns the result.
    pub async fn run(
        &self,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, crate::error::Error> {
        match self.transport_type {
            McpTransportType::HttpStream | McpTransportType::Sse => {
                self.run_http(input).await
            }
            McpTransportType::Stdio => self.run_stdio(input).await,
        }
    }

    async fn run_http(
        &self,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, crate::error::Error> {
        let client = reqwest::Client::new();
        let url = format!("{}/mcp/tools/call", self.endpoint.trim_end_matches('/'));

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": self.name,
                "arguments": input
            },
            "id": 1
        });

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("HTTP request failed: {e}")))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to parse response: {e}")))?;

        extract_tool_result(&body)
    }

    async fn run_stdio(
        &self,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, crate::error::Error> {
        use tokio::process::Command;
        use tokio::io::AsyncWriteExt;

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
            "method": "tools/call",
            "params": {
                "name": self.name,
                "arguments": input
            },
            "id": 1
        });

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());

        let mut child = cmd
            .spawn()
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to spawn process: {e}")))?;

        if let Some(stdin) = child.stdin.as_mut() {
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
        let body: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to parse output: {e}")))?;

        extract_tool_result(&body)
    }
}

/// Creates MCP tools from an endpoint by fetching definitions.
///
/// Convenience function that connects to an MCP server, lists available tools,
/// and returns `McpTool` instances for each.
///
/// Mirrors Python's `MCPFactory.create_tools()`.
pub async fn create_mcp_tools(
    endpoint: impl Into<String>,
    transport_type: McpTransportType,
) -> Result<Vec<McpTool>, crate::error::Error> {
    let endpoint = endpoint.into();
    let service = McpDefinitionService::new(&endpoint, transport_type.clone());
    let definitions = service.fetch_tool_definitions().await?;

    Ok(definitions
        .into_iter()
        .map(|def| McpTool::new(def, &endpoint, transport_type.clone()))
        .collect())
}

fn extract_tool_result(
    body: &serde_json::Value,
) -> Result<serde_json::Value, crate::error::Error> {
    // Check for JSON-RPC error
    if let Some(error) = body.get("error") {
        let message = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown MCP error");
        return Err(crate::error::Error::Mcp(message.to_string()));
    }

    // Extract result
    let result = body
        .get("result")
        .ok_or_else(|| crate::error::Error::Mcp("No result in response".to_string()))?;

    // MCP tool results typically have a "content" array
    if let Some(content) = result.get("content") {
        if let Some(arr) = content.as_array() {
            // If single text content, extract just the text
            if arr.len() == 1 {
                if let Some(text) = arr[0].get("text").and_then(|t| t.as_str()) {
                    // Try to parse as JSON for structured output
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
                        return Ok(parsed);
                    }
                    return Ok(serde_json::Value::String(text.to_string()));
                }
            }
            return Ok(content.clone());
        }
    }

    Ok(result.clone())
}

impl std::fmt::Debug for McpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("transport_type", &self.transport_type)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_result_text() {
        let body = serde_json::json!({
            "result": {
                "content": [
                    {"type": "text", "text": "Hello, world!"}
                ]
            }
        });
        let result = extract_tool_result(&body).unwrap();
        assert_eq!(result, serde_json::Value::String("Hello, world!".to_string()));
    }

    #[test]
    fn test_extract_tool_result_json() {
        let body = serde_json::json!({
            "result": {
                "content": [
                    {"type": "text", "text": "{\"key\": \"value\"}"}
                ]
            }
        });
        let result = extract_tool_result(&body).unwrap();
        assert_eq!(result, serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_extract_tool_result_error() {
        let body = serde_json::json!({
            "error": {
                "code": -32600,
                "message": "Invalid request"
            }
        });
        let result = extract_tool_result(&body);
        assert!(result.is_err());
    }

    #[test]
    fn test_mcp_tool_creation() {
        let def = McpToolDefinition {
            name: "search".to_string(),
            description: Some("Search the web".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
            output_schema: None,
        };
        let tool = McpTool::new(def, "http://localhost:8080", McpTransportType::HttpStream);
        assert_eq!(tool.name, "search");
        assert_eq!(tool.description.as_deref(), Some("Search the web"));
    }
}
