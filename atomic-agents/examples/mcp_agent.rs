//! MCP agent example demonstrating dynamic tool creation from an MCP server.
//!
//! Usage:
//!   Start an MCP server, then:
//!   OPENAI_API_KEY=sk-... MCP_ENDPOINT=http://localhost:8080 cargo run --example mcp_agent

use atomic_agents::mcp::{McpDefinitionService, McpTransportType, McpTool};
use atomic_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let _api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let mcp_endpoint =
        std::env::var("MCP_ENDPOINT").unwrap_or_else(|_| "http://localhost:8080".to_string());

    // Discover available tools from the MCP server
    println!("Connecting to MCP server at {mcp_endpoint}...");
    let service = McpDefinitionService::new(&mcp_endpoint, McpTransportType::HttpStream);

    match service.fetch_tool_definitions().await {
        Ok(definitions) => {
            println!("Found {} tools:", definitions.len());
            for def in &definitions {
                println!(
                    "  - {} ({})",
                    def.name,
                    def.description.as_deref().unwrap_or("no description")
                );
            }

            // Create McpTool instances
            let tools: Vec<McpTool> = definitions
                .into_iter()
                .map(|def| McpTool::new(def, &mcp_endpoint, McpTransportType::HttpStream))
                .collect();

            // Use the first tool if available
            if let Some(tool) = tools.first() {
                println!("\nCalling tool '{}'...", tool.name);
                let result = tool
                    .run(serde_json::json!({"query": "test"}))
                    .await?;
                println!("Result: {result}");
            }
        }
        Err(e) => {
            println!("Could not connect to MCP server: {e}");
            println!("Make sure an MCP server is running at {mcp_endpoint}");
        }
    }

    Ok(())
}
