# Changelog

## Unreleased — next minor (0.3.0)

### Breaking (mcp feature)

- `McpTool::new(definition, endpoint, transport)` and `with_working_directory` are removed; tools come from `McpClient::tools()` or `McpTool::from_definition(def, client)`.
- `McpTool::run(Value)` is replaced by `McpTool::call(Map)` (async) and `Tool::run(McpToolInput)` (sync); the public fields `name`, `description`, `input_schema` and `output_schema` become accessors.
- `create_mcp_tools(endpoint, transport_type) -> Result<Vec<McpTool>>` now returns `Result<(McpClient, Vec<McpTool>)>`.
- `McpDefinitionService` keeps `new`, `with_working_directory`, `fetch_tool_definitions`, `fetch_resource_definitions` and `fetch_prompt_definitions`, but each call now opens and closes a real session; HTTP endpoints resolve to `/mcp` instead of `/mcp/<method>`.
- `McpResourceDefinition` gains `is_template`; `McpPromptDefinition` gains `arguments`. `McpPaginationError::MissingItems` is removed.
- `McpTransportType` is unchanged.
- `atomic_agents::mcp` is now compiled only with the `mcp` feature; before, it was compiled unconditionally.

### Breaking (core)

- `AgentConfig` gains a public `output_schema: Option<serde_json::Value>` field. It is not `#[non_exhaustive]`, so struct-literal construction must add the field (or use `AgentConfig::new`).

### Added

- `McpClient`: a persistent, initialised MCP session over stdio, Streamable HTTP or legacy HTTP+SSE, built on `rmcp`. Discovery is capability-gated and paginated, with repeated-cursor detection.
- `McpTransport` (`stdio`, `streamable_http`, `sse`, `with_cwd`, `with_env`, `with_header`, `from_endpoint`); `Authorization: Bearer` headers are supported.
- `McpResource` and `McpPrompt` implementing `Resource` and `Prompt`, with `McpResourceOutput` and `McpPromptOutput`.
- Orchestrator support: `orchestrator_schema`, `McpToolset`, `McpToolCall`, `McpOrchestratorOutput`, and `fetch_mcp_*` helpers.
- `AgentConfig::with_output_schema` to override the output JSON Schema sent to the model.
- Explicit `Error::Mcp` messages for unsupported `input_required` and task replies.
- URL userinfo, query and fragment are redacted from MCP error text; header values are marked sensitive.
- Legacy SSE hardening: same-origin POST endpoint, no redirects, 30 s connect budget, 16 MiB per-event cap.
- The orchestrator schema does not require `serde_json/preserve_order`; `tool_name` is listed first in `required` only.

### Migration

```rust
// before
let tool = McpTool::new(def, endpoint, kind);

// after
let client = McpClient::connect_endpoint(endpoint, kind).await?;
let tools = client.tools().await?;
```
