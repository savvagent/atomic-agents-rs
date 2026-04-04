//! Basic chatbot example using OpenAI.
//!
//! Usage: OPENAI_API_KEY=sk-... cargo run --example basic_chatbot

use atomic_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let client = OpenAiClient::new(api_key);
    let config = AgentConfig::new(client, "gpt-4o-mini");
    let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

    let response = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: "What is the capital of France?".to_string(),
        }))
        .await?;

    println!("Agent: {}", response.chat_message);

    // Multi-turn conversation
    let response = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: "What about Germany?".to_string(),
        }))
        .await?;

    println!("Agent: {}", response.chat_message);

    Ok(())
}
