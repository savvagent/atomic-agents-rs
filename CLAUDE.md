# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust port of the [atomic-agents](https://github.com/BrainBlend-AI/atomic-agents) Python framework — a lightweight, modular framework for building AI agent pipelines with structured LLM outputs. Uses Rust edition 2024.

## Build & Test Commands

```bash
cargo build                              # Build all workspace crates
cargo test                               # Run all tests
cargo test -p atomic-agents              # Test core crate only
cargo test -p atomic-forge               # Test tools crate only
cargo test test_agent_run_async          # Run a single test by name
cargo build -p atomic-agents --features "openai,anthropic,google,mcp"  # All features
cargo run -p atomic-assembler            # Run the TUI tool browser
cargo run --example basic_chatbot -p atomic-agents --features openai   # Run an example
```

Feature flags on `atomic-agents`: `openai` (default), `anthropic`, `google`, `mcp`.

## Workspace Structure

Four crates in a Cargo workspace:

- **`atomic-agents`** — Core framework: agent, LLM clients, schemas, chat history, streaming, hooks, MCP integration
- **`atomic-agents-macros`** — Proc macro crate providing `#[derive(IOSchema)]`
- **`atomic-forge`** — Collection of reusable tools (calculator, web search, scraper, YouTube transcripts)
- **`atomic-assembler`** — TUI application (ratatui) for browsing and downloading atomic-forge tools

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

Behind the `mcp` feature flag. Uses `serde_json::Value` for dynamic schemas since Rust can't create types at runtime. `McpDefinitionService` fetches definitions; `McpTool` wraps them as dynamic tool implementations.

### SystemPromptGenerator

Builder pattern with sections: background, steps, output instructions. Supports `ContextProvider` trait for injecting dynamic context at runtime.
