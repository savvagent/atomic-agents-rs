//! Orchestration example demonstrating tool selection via structured output.
//!
//! Usage: OPENAI_API_KEY=sk-... cargo run --example orchestration

use atomic_agents::prelude::*;
use atomic_forge::calculator::{CalculatorInput, CalculatorTool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The orchestrator decides which tool to use based on user input.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct OrchestratorOutput {
    /// The tool to use: "calculator" or "none".
    pub tool: String,
    /// If tool is "calculator", the expression to evaluate.
    pub expression: Option<String>,
    /// If no tool is needed, the direct response.
    pub direct_response: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let client = OpenAiClient::new(api_key);
    let system_prompt = SystemPromptGenerator::new().with_background(vec![
        "You are an assistant that can perform calculations.".to_string(),
        "If the user asks a math question, set tool to 'calculator' and provide the expression."
            .to_string(),
        "Otherwise, respond directly with direct_response.".to_string(),
    ]);

    let config = AgentConfig::new(client, "gpt-4o-mini").with_system_prompt(system_prompt);
    let agent = AtomicAgent::<BasicChatInputSchema, OrchestratorOutput, _>::new(config);

    let calc = CalculatorTool::default();

    // Test with a math question
    let response = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: "What is the square root of 144?".to_string(),
        }))
        .await?;

    println!("Orchestrator decided: tool={}", response.tool);

    if response.tool == "calculator" {
        if let Some(expr) = &response.expression {
            let result = calc.run(CalculatorInput {
                expression: expr.clone(),
            })?;
            println!("Calculator result: {}", result.result);
        }
    } else if let Some(resp) = &response.direct_response {
        println!("Direct response: {resp}");
    }

    Ok(())
}
