//! Default `ClientFactory` implementation that pulls API keys from environment
//! variables. Used by the HTTP server.

use atomic_agents::client::LlmClient;

use crate::error::Error;
use crate::interpreter::ClientFactory;

/// Reads API keys from environment variables:
///
/// - `OPENAI_API_KEY`
/// - `ANTHROPIC_API_KEY`
/// - `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
pub struct EnvClientFactory;

impl ClientFactory for EnvClientFactory {
    fn create_client(&self, provider: &str) -> Result<Box<dyn LlmClient>, Error> {
        match provider {
            "openai" => {
                let key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                    Error::Pipeline("OPENAI_API_KEY environment variable is not set".into())
                })?;
                Ok(Box::new(atomic_agents::client::openai::OpenAiClient::new(key)))
            }
            "anthropic" => {
                let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                    Error::Pipeline("ANTHROPIC_API_KEY environment variable is not set".into())
                })?;
                Ok(Box::new(atomic_agents::client::anthropic::AnthropicClient::new(key)))
            }
            "google" => {
                let key = std::env::var("GOOGLE_API_KEY")
                    .or_else(|_| std::env::var("GEMINI_API_KEY"))
                    .map_err(|_| {
                        Error::Pipeline(
                            "GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable is not set"
                                .into(),
                        )
                    })?;
                Ok(Box::new(atomic_agents::client::google::GoogleClient::new(key)))
            }
            other => Err(Error::Pipeline(format!(
                "Unknown or unsupported provider '{other}'"
            ))),
        }
    }
}
