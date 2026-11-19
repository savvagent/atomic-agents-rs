//! Connects to an MCP server over Streamable HTTP and lists its tools.
//!
//! Usage:
//!   MCP_ENDPOINT=http://localhost:8080 cargo run -p atomic-agents --features mcp --example mcp_agent

use atomic_agents::error::Result;
use atomic_agents::mcp::{McpClient, McpTransportType};

#[tokio::main]
async fn main() -> Result<()> {
    let endpoint =
        std::env::var("MCP_ENDPOINT").unwrap_or_else(|_| "http://localhost:8080".to_string());
    println!("Connecting to MCP server at {endpoint}...");
    let client = McpClient::connect_endpoint(&endpoint, McpTransportType::HttpStream).await?;
    let tools = client.list_tools().await?;
    println!("Found {} tools:", tools.len());
    for tool in &tools {
        println!(
            "  - {} ({})",
            tool.name,
            tool.description.as_deref().unwrap_or("no description")
        );
    }
    client.close().await
}
