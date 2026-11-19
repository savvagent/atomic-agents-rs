# Atomic Agents (Rust)

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

## What is Atomic Agents?

Atomic Agents is a lightweight, modular framework for building AI agent pipelines with structured LLM outputs. This is the **Rust port** of the [Python Atomic Agents framework](https://github.com/Eigenwise/atomic-agents).

Think of it like building AI applications with LEGO blocks — each component (agent, tool, context provider) is:
- **Single-purpose**: Does one thing well
- **Reusable**: Can be used in multiple pipelines
- **Composable**: Easily combines with other components
- **Predictable**: Produces consistent, reliable outputs

Built on [serde](https://serde.rs/) and [schemars](https://graham.cool/schemars/) for structured I/O, it brings the same principles as the Python version with Rust's type safety and performance.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quick Example](#quick-example)
- [Why Atomic Agents?](#why-atomic-agents)
- [Core Concepts](#core-concepts)
  - [Anatomy of an Agent](#anatomy-of-an-agent)
  - [Context Providers](#context-providers)
  - [Chaining Schemas and Agents](#chaining-schemas-and-agents)
  - [MCP Tools](#mcp-tools)
- [Examples](#examples)
- [Atomic Forge & CLI](#atomic-forge--cli)
- [Project Structure](#project-structure)
- [Provider & Model Compatibility](#provider--model-compatibility)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Installation

Add `atomic-agents` to your `Cargo.toml`:

```toml
[dependencies]
atomic-agents = "0.2"
```

By default, the OpenAI provider is enabled. Enable additional providers with feature flags:

```toml
[dependencies]
atomic-agents = { version = "0.2", features = ["openai", "anthropic", "google", "mcp"] }
```

### Quick Example

```rust
use atomic_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAiClient::new("sk-...");
    let config = AgentConfig::new(client, "gpt-4o-mini");
    let agent = AtomicAgent::<BasicChatInputSchema, BasicChatOutputSchema, _>::new(config);

    let response = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: "What is the capital of France?".to_string(),
        }))
        .await?;

    println!("Agent: {}", response.chat_message);
    Ok(())
}
```

## Why Atomic Agents?

While existing frameworks for agentic AI focus on building autonomous multi-agent systems, they often lack the control and predictability required for real-world applications. Businesses need AI systems that produce consistent, reliable outputs aligned with their brand and objectives.

Atomic Agents addresses this need by providing:

- **Modularity:** Build AI applications by combining small, reusable components.
- **Predictability:** Define clear input and output schemas to ensure consistent behavior.
- **Extensibility:** Easily swap out components or integrate new ones without disrupting the entire system.
- **Control:** Fine-tune each part of the system individually, from system prompts to tool integrations.

## Core Concepts

### Anatomy of an Agent

In Atomic Agents, an agent is composed of several key components:

- **System Prompt:** Defines the agent's behavior and purpose.
- **Input Schema:** Specifies the structure and validation rules for the agent's input.
- **Output Schema:** Specifies the structure and validation rules for the agent's output.
- **History:** Stores conversation history.
- **Context Providers:** Inject dynamic context into the agent's system prompt at runtime.

### Custom Output Schemas

Define structured outputs with `#[derive(IOSchema)]`. A doc comment is required and serves as the schema description:

```rust
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
}

let agent = AtomicAgent::<BasicChatInputSchema, CityAnalysisOutput, _>::new(config);
```

### Context Providers

Enhance your agents with dynamic context using the `ContextProvider` trait. Context providers inject additional information into the agent's system prompt at runtime:

```rust
use atomic_agents::prelude::*;

struct SearchResultsProvider {
    results: Vec<String>,
}

impl ContextProvider for SearchResultsProvider {
    fn title(&self) -> &str {
        "Search Results"
    }

    fn get_info(&self) -> String {
        self.results.join("\n")
    }
}

// Register the context provider with the agent
agent.register_context_provider(
    "search_results",
    Box::new(SearchResultsProvider {
        results: vec!["Result 1".into(), "Result 2".into()],
    }),
);
```

### Chaining Schemas and Agents

Chain agents and tools by aligning their input and output schemas. When one agent's output schema matches a tool's input schema, the output can be fed directly as input — enabling modular, swappable pipelines.

### MCP Tools

With the `mcp` feature, an `McpClient` holds one persistent, initialised session with an [MCP](https://modelcontextprotocol.io) server. Its tools implement `Tool`, and `McpToolset` merges them into a single output schema so an agent can pick one:

```rust
use atomic_agents::mcp::{McpOrchestratorOutput, McpToolset, McpTransportType, fetch_mcp_attributes_with_schema};
use atomic_agents::prelude::*;

let (client, tools, schema) =
    fetch_mcp_attributes_with_schema("http://localhost:8080", McpTransportType::HttpStream).await?;
let toolset = McpToolset::new(tools)?;

let config = AgentConfig::new(OpenAiClient::new(api_key), "gpt-4o-mini")
    .with_output_schema(schema);
let agent = AtomicAgent::<BasicChatInputSchema, McpOrchestratorOutput, _>::new(config);

let choice = agent.run_async(Some(BasicChatInputSchema { chat_message: "add 2 and 3".into() })).await?;
let result = toolset.dispatch(&choice.tool_parameters).await?;
client.close().await?;
```

Resources and prompts are available through `client.resources()` and `client.prompts()`; discovery follows pagination and only asks for what the server advertises.

| Transport | `McpTransportType` | Endpoint string |
|-----------|--------------------|-----------------|
| Streamable HTTP (recommended) | `HttpStream` | URL; `/mcp` is appended unless the path already ends with it |
| Legacy HTTP + SSE (2024-11-05) | `Sse` | URL; `/sse` is appended unless the path already ends with it |
| stdio | `Stdio` | Shell-quoted command line, e.g. `npx -y my-server` |

Use `McpTransport` directly for full control (`with_cwd`, `with_env`, `with_header`). `Authorization: Bearer <token>` headers are supported on both HTTP transports.

Notes:

- `McpTool::run` blocks, so it needs a multi-threaded tokio runtime; use `McpTool::call` (async) otherwise.
- `input_required` and task replies from a server are reported as explicit `Error::Mcp` messages; they are not supported.
- URL userinfo, query strings and fragments are redacted from error text.
- HTTPS uses native-tls (OpenSSL on Linux, the platform TLS stack elsewhere), the same backend as the crate's other HTTP clients; the `mcp` feature adds no rustls.
- The legacy SSE transport only accepts a same-origin POST endpoint, never follows redirects, allows 30 s to connect and caps each event at 16 MiB.
- The orchestrator schema is deterministic (sorted keys) and lists `tool_name` first in `required`; it does not rely on property order.

## Examples

Examples are in the `atomic-agents/examples/` directory. Run them with:

```bash
OPENAI_API_KEY=sk-... cargo run --example <name> -p atomic-agents
```

Available examples:

- **[basic_chatbot](atomic-agents/examples/basic_chatbot.rs)** — Minimal chatbot with multi-turn conversation
- **[custom_schema](atomic-agents/examples/custom_schema.rs)** — Structured output with a custom schema
- **[streaming_chatbot](atomic-agents/examples/streaming_chatbot.rs)** — Streaming responses as they arrive
- **[hooks](atomic-agents/examples/hooks.rs)** — Hook system for monitoring and error handling
- **[multi_provider](atomic-agents/examples/multi_provider.rs)** — Using different LLM providers (OpenAI, Anthropic, Google)
- **[orchestration](atomic-agents/examples/orchestration.rs)** — Orchestrator agent that routes between tools
- **[mcp_agent](atomic-agents/examples/mcp_agent.rs)** — Agent that selects and calls a tool from an MCP server (`MCP_ENDPOINT`, `MCP_TRANSPORT`, `OPENAI_API_KEY`)

Some examples require additional feature flags:

```bash
# Anthropic
ANTHROPIC_API_KEY=... cargo run --example multi_provider -p atomic-agents --features anthropic

# MCP
OPENAI_API_KEY=... MCP_ENDPOINT=http://localhost:8080 cargo run --example mcp_agent -p atomic-agents --features mcp
```

## Atomic Forge & CLI

**Atomic Forge** is a collection of reusable tools that can be used with Atomic Agents:

- Calculator — Mathematical expression evaluation
- Tavily Search — Web search via Tavily API
- SearXNG Search — Meta-search via SearXNG
- BoCha Search — Web search via BoCha API
- arXiv Search — Academic paper search via the arXiv API
- Hacker News Search — Stories and comments via Algolia
- Wikipedia Search — Encyclopedic article summaries
- Weather — Current conditions and forecast via Open-Meteo
- Firecrawl Search — Web and news search via Firecrawl
- DateTime — Time zones, parsing, shifting and diffs (feature `datetime`)
- PDF Reader — Text and metadata extraction (feature `pdf-reader`)
- Webpage Scraper — HTML to text scraping
- YouTube Transcript — Video transcript extraction

**Atomic Assembler** is a TUI application for browsing and downloading tools:

```bash
cargo run -p atomic-assembler
```

## Project Structure

Atomic Agents uses a Cargo workspace with the following crates:

1. **`atomic-agents/`** — Core framework: agents, LLM clients, schemas, chat history, streaming, hooks
2. **`atomic-agents-macros/`** — Proc macro crate providing `#[derive(IOSchema)]`
3. **`atomic-forge/`** — Collection of reusable tools for use with agents
4. **`atomic-assembler/`** — TUI application for browsing and downloading tools

For local development:

```bash
git clone https://github.com/savvagent/atomic-agents-rs.git
cd atomic-agents-rs
cargo build
cargo test
```

## Provider & Model Compatibility

Atomic Agents supports multiple LLM providers through feature flags:

| Provider | Feature Flag | Client | Notes |
|----------|-------------|--------|-------|
| OpenAI | `openai` (default) | `OpenAiClient` | Also works with OpenAI-compatible APIs |
| Anthropic | `anthropic` | `AnthropicClient` | Claude models |
| Google | `google` | `GoogleClient` | Gemini models |

Implement the `LlmClient` trait to add support for additional providers.

## Contributing

We welcome contributions! Here are the quick steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Run tests (`cargo test`)
5. Format your code (`cargo fmt`)
6. Lint your code (`cargo clippy`)
7. Commit your changes (`git commit -m 'Add some feature'`)
8. Push to the branch (`git push origin feature-branch`)
9. Open a pull request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
