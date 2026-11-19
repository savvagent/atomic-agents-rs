# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust port of the [atomic-agents](https://github.com/Eigenwise/atomic-agents) Python framework — a lightweight, modular framework for building AI agent pipelines with structured LLM outputs. Uses Rust edition 2024.

## Build & Test Commands

```bash
cargo build                              # Build all workspace crates
cargo test                               # Run all tests
cargo test -p atomic-agents              # Test core crate only
cargo test -p atomic-forge               # Test tools crate only
cargo test test_agent_run_async          # Run a single test by name
cargo test -p atomic-agents --features mcp --test mcp_integration  # MCP integration scenarios (harness=false)
cargo build -p atomic-agents --features "openai,anthropic,google,mcp"  # All features
cargo run -p atomic-assembler            # Run the TUI tool browser
cargo run --example basic_chatbot -p atomic-agents --features openai   # Run an example
cargo run -p atomic-studio-runner -- serve --addr 127.0.0.1:8787 \
  --static-dir atomic-studio/build       # Serve the studio PWA + JSON API
```

Feature flags on `atomic-agents`: `openai` (default), `anthropic`, `google`, `mcp`.

### atomic-studio (PWA)

The studio is a SvelteKit + Svelte Flow PWA living in `atomic-studio/`. It is a
local-first installable web app — pipelines persist to IndexedDB, can be
imported/exported as JSON, and execute against `atomic-studio-runner`'s HTTP API.
The PWA uses SvelteKit's built-in service worker (`src/service-worker.ts`) — no
third-party PWA plugin.

```bash
cd atomic-studio && pnpm install && pnpm dev   # Dev server (proxies /api → localhost:8787)
cd atomic-studio && pnpm build                  # Build static assets to atomic-studio/build/
```

The runner's `serve` subcommand exposes:

- `POST /api/validate` — JSON pipeline → `{ valid, errors, warnings }`
- `POST /api/runs`     — `{ pipeline, input }` → newline-delimited JSON stream of
  `ExecutionEvent`s over chunked HTTP transfer (consumed via fetch ReadableStream).
- Static fallback serves `--static-dir` (typically `atomic-studio/build`) so the
  PWA and API live behind a single origin.

API keys for agent execution come from environment variables: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).

## Workspace Structure

Five crates in a Cargo workspace:

- **`atomic-agents`** — Core framework: agent, LLM clients, schemas, chat history, streaming, hooks, MCP integration
- **`atomic-agents-macros`** — Proc macro crate providing `#[derive(IOSchema)]`
- **`atomic-forge`** — Collection of reusable tools (calculator, web search, scraper, YouTube transcripts)
- **`atomic-assembler`** — TUI application (ratatui) for browsing and downloading atomic-forge tools
- **`atomic-studio-runner`** — Pipeline interpreter + HTTP server backing the atomic-studio PWA

## Architecture

### IOSchema trait + derive macro

The `IOSchema` trait (`atomic-agents/src/schema.rs`) is the foundation — all agent inputs/outputs must implement it. The `#[derive(IOSchema)]` proc macro (`atomic-agents-macros/src/lib.rs`) generates the implementation, requiring a doc comment (used as schema description). Structs must also derive `Serialize`, `Deserialize`, `JsonSchema`, `Clone`, `Debug`.

### Agent (`AtomicAgent<I, O, C>`)

Generic over input schema `I: IOSchema`, output schema `O: IOSchema`, and client `C: LlmClient`. Three execution modes: `run()` (sync), `run_async()`, `run_async_stream()`. Output extraction uses `ExtractionMode`: `Tools` (function calling, default), `Json`, or `MdJson`. Failed JSON parsing triggers automatic retries with error context appended to the conversation.

### LLM Clients (`LlmClient` trait)

Feature-gated provider implementations behind `async_trait`: `OpenAiClient`, `AnthropicClient`, `GoogleClient`. Each handles provider-specific API formats, role naming, and SSE stream parsing. A `MockClient` exists in `#[cfg(test)]` for unit tests.

### Tool trait

`Tool` trait (`atomic-agents/src/tool.rs`) defines `Input`/`Output` associated types (both `IOSchema`) and a `run()` method. Tools in `atomic-forge` implement this trait.

### MCP integration

Behind the `mcp` feature flag (the `mcp` module is not compiled without it). Built on `rmcp`. `McpClient` = one persistent, initialised session (stdio child / Streamable HTTP / legacy SSE). Discovery is capability-gated and paginated with repeated-cursor detection. `McpTool`/`McpResource`/`McpPrompt` implement `Tool`/`Resource`/`Prompt` (sync `run` needs a multi-thread runtime; async `call`/`read_async`/`get`). `orchestrator_schema` + `AgentConfig::with_output_schema` let an agent pick a tool; `McpToolset::dispatch` runs the choice. `McpDefinitionService` is a one-shot compat wrapper. Integration tests are in `atomic-agents/tests/mcp_integration.rs` (harness=false, re-execs itself as the stdio server) — run with `cargo test -p atomic-agents --features mcp --test mcp_integration`.

### SystemPromptGenerator

Builder pattern with sections: background, steps, output instructions. Supports `ContextProvider` trait for injecting dynamic context at runtime.
