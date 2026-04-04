//! Hooks example demonstrating the event system.
//!
//! Usage: OPENAI_API_KEY=sk-... cargo run --example hooks

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atomic_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let client = OpenAiClient::new(api_key);
    let config = AgentConfig::new(client, "gpt-4o-mini");
    let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

    // Track completion events
    let completion_count = Arc::new(AtomicUsize::new(0));
    let count_clone = completion_count.clone();
    let on_completion: HookHandler = Arc::new(move |data| {
        let n = count_clone.fetch_add(1, Ordering::SeqCst) + 1;
        println!("[Hook] Completion #{n} received: {data}");
    });
    agent.register_hook(HookEvent::CompletionResponse, on_completion);

    // Track errors
    let on_error: HookHandler = Arc::new(|data| {
        println!("[Hook] Parse error: {data}");
    });
    agent.register_hook(HookEvent::ParseError, on_error);

    // Run a few queries
    for question in &["Hello!", "What is Rust?", "Goodbye!"] {
        let response = agent
            .run_async(Some(BasicChatInputSchema {
                chat_message: question.to_string(),
            }))
            .await?;
        println!("Agent: {}", response.chat_message);
        println!("---");
    }

    println!(
        "Total completions: {}",
        completion_count.load(Ordering::SeqCst)
    );

    Ok(())
}
