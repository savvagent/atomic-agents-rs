//! Streaming chatbot example using OpenAI.
//!
//! Usage: OPENAI_API_KEY=sk-... cargo run --example streaming_chatbot

use atomic_agents::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let client = OpenAiClient::new(api_key);
    let config = AgentConfig::new(client, "gpt-4o-mini");
    let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

    print!("Agent: ");

    let mut stream = agent
        .run_async_stream(Some(BasicChatInputSchema {
            chat_message: "Tell me a short joke about programming.".to_string(),
        }))
        .await?;

    while let Some(partial) = stream.next().await {
        let partial = partial?;
        // Print the raw accumulated content
        print!("\rAgent: {}", partial.raw);
    }
    println!();

    Ok(())
}
