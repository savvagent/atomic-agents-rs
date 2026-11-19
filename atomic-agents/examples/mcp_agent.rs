//! An agent that picks one tool from an MCP server and calls it.
//!
//! The server's tools are merged into a single output schema; the model
//! answers with the tool it wants and its arguments, and the toolset
//! dispatches that call over the same MCP session.
//!
//! Usage:
//!   OPENAI_API_KEY=sk-... MCP_ENDPOINT=http://localhost:8080 \
//!     cargo run -p atomic-agents --features mcp --example mcp_agent
//!
//! `MCP_TRANSPORT` selects `http` (default), `sse` or `stdio`; for `stdio`,
//! `MCP_ENDPOINT` is the server command line.

use atomic_agents::error::{Error, Result};
use atomic_agents::mcp::{
    McpOrchestratorOutput, McpToolset, McpTransportType, fetch_mcp_attributes_with_schema,
};
use atomic_agents::prelude::*;

fn transport_from_env() -> Result<McpTransportType> {
    match std::env::var("MCP_TRANSPORT").as_deref() {
        Err(_) | Ok("http") => Ok(McpTransportType::HttpStream),
        Ok("sse") => Ok(McpTransportType::Sse),
        Ok("stdio") => Ok(McpTransportType::Stdio),
        Ok(other) => Err(Error::Config(format!(
            "MCP_TRANSPORT must be http, sse or stdio, got {other:?}"
        ))),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| Error::Config("OPENAI_API_KEY must be set".into()))?;
    let endpoint =
        std::env::var("MCP_ENDPOINT").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let kind = transport_from_env()?;

    println!("Connecting to MCP server at {endpoint}...");
    let (client, tools, schema) = fetch_mcp_attributes_with_schema(&endpoint, kind).await?;
    println!("Found {} tools", tools.len());
    let toolset = McpToolset::new(tools)?;

    let config =
        AgentConfig::new(OpenAiClient::new(api_key), "gpt-4o-mini").with_output_schema(schema);
    let agent = AtomicAgent::<BasicChatInputSchema, McpOrchestratorOutput, _>::new(config);

    let question = "Use one of the available tools to help me with a simple task.";
    println!("User: {question}");
    let choice = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: question.to_string(),
        }))
        .await?;
    println!(
        "Agent chose '{}' with {:?}",
        choice.tool_parameters.tool_name, choice.tool_parameters.arguments
    );

    let result = toolset.dispatch(&choice.tool_parameters).await?;
    println!("Result: {result}");

    client.close().await
}
