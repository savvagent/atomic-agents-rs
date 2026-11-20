//! Custom schema example demonstrating structured output.
//!
//! Usage: OPENAI_API_KEY=sk-... cargo run --example custom_schema

use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Analyze a city and return structured information about it.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct CityAnalysisOutput {
    /// The name of the city.
    pub city: String,
    /// The country the city is in.
    pub country: String,
    /// The approximate population.
    pub population: String,
    /// The top 3 attractions.
    pub top_attractions: Vec<String>,
    /// A brief one-sentence description.
    pub description: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let client = OpenAiClient::new(api_key);
    let system_prompt = SystemPromptGenerator::new().with_background(vec![
        "You are a knowledgeable travel guide.".to_string(),
        "You provide accurate, structured information about cities.".to_string(),
    ]);

    let config = AgentConfig::new(client, "gpt-4o-mini").with_system_prompt(system_prompt);

    let agent = AtomicAgent::<BasicChatInputSchema, CityAnalysisOutput, _>::new(config);

    let result = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: "Tell me about Tokyo".to_string(),
        }))
        .await?;

    println!("City: {}", result.city);
    println!("Country: {}", result.country);
    println!("Population: {}", result.population);
    println!("Description: {}", result.description);
    println!("Top Attractions:");
    for (i, attraction) in result.top_attractions.iter().enumerate() {
        println!("  {}. {}", i + 1, attraction);
    }

    Ok(())
}
