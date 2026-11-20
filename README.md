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
- [Running agents in a WebAssembly component](#running-agents-in-a-webassembly-component)
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

Use `McpTransport` directly for full control (`with_cwd`, `with_env`, `with_inherited_env`, `with_header`, `with_request_timeout`, `with_connect_timeout`). `Authorization: Bearer <token>` headers are supported on both HTTP transports.

**Trust boundary:** a connected MCP server is inside your trust boundary. Its tool names, descriptions and schemas reach the LLM as-is and unmarked, so a malicious server can steer the model. `McpToolset::dispatch` only calls tools discovered from the server, but arguments are not validated against a tool's `input_schema`. Connect only to servers you trust.

Notes:

- `McpTool::run` blocks, so it needs a multi-threaded tokio runtime; use `McpTool::call` (async) otherwise.
- A stdio server gets only a small allowlist of the parent's environment variables (`PATH`, `HOME`, `USER`, ... — see `DEFAULT_INHERITED_ENV_VARS`) plus those set with `with_env`, so API keys in your environment do not reach it. Use `with_inherited_env()` to pass everything.
- Each request (every list page, tool call, resource read and prompt render) times out after 60 s by default, and connecting plus `initialize` after 30 s; a timed-out request is cancelled and the session stays usable. Lists stop with an error after 1000 pages.
- `input_required` and task replies from a server are reported as explicit `Error::Mcp` messages; they are not supported.
- URL userinfo, query strings and fragments are redacted from error text, and HTTP client errors from the Streamable HTTP transport carry no URL, so rmcp's own `tracing` output does not leak them either.
- HTTPS uses native-tls (OpenSSL on Linux, the platform TLS stack elsewhere), the same backend as the crate's other HTTP clients; the `mcp` feature adds no rustls.
- The legacy SSE transport only accepts a same-origin POST endpoint, never follows redirects, allows the connect timeout (30 s by default) to connect and caps each event at 16 MiB.
- The orchestrator schema is deterministic (sorted keys) and lists `tool_name` first in `required`; it does not rely on property order.
- The orchestrator schema drops each tool's own `$schema` and hoists its `$defs` into the root. Gemini function declarations do not support `$ref`/`$defs`, so tools whose schemas use them may need flattening for Google models.

## Running agents in a WebAssembly component

`atomic-agents` has two feature groups that decide what it depends on:

- `native` (on by default through `openai`; implied by `openai`, `anthropic`, `google` and `mcp`) carries the network and runtime dependencies: `tokio`, `reqwest`, `tiktoken-rs`, `bytes`, `eventsource-stream` and `tokio-stream`. `ClientError::Http` exists only with `native`.
- `guest` is pure Rust: host traits and adapters (`atomic_agents::guest`) that let an agent call a model and tools provided by the embedding host instead of opening connections itself. It builds for WebAssembly components:

```bash
rustup target add wasm32-wasip2
cargo build -p atomic-agents --no-default-features --features guest --target wasm32-wasip2
```

Without `native` and without a provider feature, `AtomicAgent::run` drives the future with `futures::executor::block_on`.

### The `atomic-agent-runner` component

`atomic-agent-runner/` is a pipeline interpreter packaged as a WebAssembly component (not published to crates.io; it is released as a component). It runs a studio pipeline spec (JSON) against an event (JSON) and returns the outcome as JSON. Its WIT package is `atomic-agents:runner@0.1.0`, world `runner` (`atomic-agent-runner/wit/world.wit`):

- imports: `llm` (`complete`), `mcp` (`call`, `list-tools`), `state` (`get`, `set`), `log` and `clock`;
- export: `run(spec, event)`.

Steps are `agent`, `tool`, `act` and `foreach` (at most 100 items, no nesting). Conditions are expressions or the studio's structured form. `act` arguments must be a literal object in the spec. Limits: `max_turns` defaults to 8 (at most 32), `max_retries` defaults to 2 (at most 5), and a run is capped at 500 tool calls and 200 LLM calls. `state` writes are committed only when the run succeeds.

```bash
cargo build -p atomic-agent-runner --release --target wasm32-wasip2
# component: target/wasm32-wasip2/release/atomic_agent_runner.wasm
cargo test -p atomic-agent-runner                # interpreter, native
cargo test -p atomic-agent-runner-host-tests     # builds the component and the env-probe fixture, runs them under wasmtime
```

The host tests need the `wasm32-wasip2` target. `ATOMIC_AGENT_RUNNER_WASM=<file>` makes them use a given runner component instead of building one; it skips the runner build only, since the sandbox test still builds its `env-probe` fixture, so the target is still required.

### What the host provides

The guest reaches the outside world only through the world's imports. A host is expected to give it:

- an empty environment, no arguments, no preopened directories and no inherited stdio;
- no network access: model calls go through `llm.complete` and tool calls through `mcp.call`;
- from `mcp.list-tools`, only the tools the host chooses to expose to the run; the guest offers the model nothing else.

The component imports only the world's interfaces plus `wasi:cli`, `wasi:io`, `wasi:clocks` and `wasi:random` (pulled in by Rust's standard library), and never `wasi:filesystem`, `wasi:sockets` or `wasi:http`; a host test checks this.

### Structured output

`completion-request` has no field that selects or forces a tool: the host decides how to obtain a reply matching `output-schema`. The Anthropic client in this repo sends `tool_choice: {"type":"auto"}` with a system instruction naming the extraction tool, because current Claude models reject a forced `tool_choice` with a 400. The model may therefore reply in text; that reply is parsed, and retried if it is invalid.

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
5. **`atomic-studio-runner/`** — Pipeline interpreter and HTTP server behind the studio
6. **`atomic-agent-runner/`** — The pipeline interpreter as a WebAssembly component (`host-tests/` runs it under wasmtime)

For local development:

```bash
git clone https://github.com/savvagent/atomic-agents-rs.git
cd atomic-agents-rs
cargo build
cargo test
```

`cargo build` and `cargo test` cover every crate except `atomic-agent-runner/host-tests`, which is left out of the workspace's `default-members` because it builds WebAssembly components. To run it too:

```bash
rustup target add wasm32-wasip2
cargo test --workspace                          # or: cargo test -p atomic-agent-runner-host-tests
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
