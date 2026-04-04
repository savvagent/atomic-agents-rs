//! Multi-provider example showing OpenAI, Anthropic, and Google clients.
//!
//! Usage:
//!   OPENAI_API_KEY=sk-... cargo run --example multi_provider --features openai
//!   ANTHROPIC_API_KEY=sk-... cargo run --example multi_provider --features anthropic
//!   GOOGLE_API_KEY=... cargo run --example multi_provider --features google

use atomic_agents::prelude::*;

async fn run_with_client(
    client: impl LlmClient,
    model: &str,
    provider: &str,
) -> Result<()> {
    let config = AgentConfig::new(client, model);
    let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

    let response = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: "What is 2 + 2? Reply in one word.".to_string(),
        }))
        .await?;

    println!("[{provider}] {model}: {}", response.chat_message);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // OpenAI
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let client = OpenAiClient::new(key);
        run_with_client(client, "gpt-4o-mini", "OpenAI").await?;
    }

    // Anthropic
    #[cfg(feature = "anthropic")]
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let client = AnthropicClient::new(key);
        run_with_client(client, "claude-sonnet-4-20250514", "Anthropic").await?;
    }

    // Google Gemini
    #[cfg(feature = "google")]
    if let Ok(key) = std::env::var("GOOGLE_API_KEY") {
        let client = GoogleClient::new(key);
        run_with_client(client, "gemini-2.0-flash", "Google").await?;
    }

    // Ollama (via OpenAI-compatible API)
    if std::env::var("USE_OLLAMA").is_ok() {
        let client = OpenAiClient::new("ollama")
            .with_base_url("http://localhost:11434/v1");
        run_with_client(client, "llama3", "Ollama").await?;
    }

    Ok(())
}
