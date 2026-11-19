# Changelog

## Unreleased — next minor (0.3.0)

### Breaking (mcp feature)

- `McpTool::new(definition, endpoint, transport)` and `with_working_directory` are removed; tools come from `McpClient::tools()` or `McpTool::from_definition(def, client)`.
- `McpTool::run(Value)` is replaced by `McpTool::call(Map)` (async) and `Tool::run(McpToolInput)` (sync); the public fields `name`, `description`, `input_schema` and `output_schema` become accessors.
- `create_mcp_tools(endpoint, transport_type) -> Result<Vec<McpTool>>` now returns `Result<(McpClient, Vec<McpTool>)>`.
- `McpDefinitionService` keeps `new`, `with_working_directory`, `fetch_tool_definitions`, `fetch_resource_definitions` and `fetch_prompt_definitions`, but each call now opens and closes a real session; HTTP endpoints resolve to `/mcp` instead of `/mcp/<method>`. `fetch_resource_definitions` now also returns resource templates, and every `fetch_*` returns an empty list when the server does not advertise that capability.
- `McpToolDefinition` gains `title`; `McpResourceDefinition` gains `is_template`; `McpPromptDefinition` gains `arguments`. None is `#[non_exhaustive]`, so struct-literal construction must add the fields. `McpPaginationError::MissingItems` is removed.
- `McpTransportType` is unchanged.
- A stdio MCP server no longer inherits the whole parent environment (which may hold LLM API keys). Like the upstream Python MCP SDK's `get_default_environment()`, the child gets only `DEFAULT_INHERITED_ENV_VARS` that are set in the parent (unix: `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER`; Windows: `APPDATA`, `HOMEDRIVE`, `HOMEPATH`, `LOCALAPPDATA`, `PATH`, `PROCESSOR_ARCHITECTURE`, `SYSTEMDRIVE`, `SYSTEMROOT`, `TEMP`, `USERNAME`, `USERPROFILE`), skipping values that start with `()`, plus anything given with `McpTransport::with_env`. This also applies to `McpDefinitionService` and `create_mcp_tools` with `McpTransportType::Stdio`. Pass the variables a server needs with `with_env`, or opt back in to the old behaviour with `McpTransport::with_inherited_env()`.
- `atomic_agents::mcp` is now compiled only with the `mcp` feature; before, it was compiled unconditionally.
- `rmcp` 3.x is now a public dependency of the `mcp` feature: `atomic_agents::mcp::ServerCapabilities` is a re-export of `rmcp::model::ServerCapabilities` (returned by `McpClient::server_capabilities`), so an rmcp major-version bump is a breaking change here too.

### Breaking (core)

- `AgentConfig` gains a public `output_schema: Option<serde_json::Value>` field. It is not `#[non_exhaustive]`, so struct-literal construction must add the field (or use `AgentConfig::new`).

### Added

- `McpClient`: a persistent, initialised MCP session over stdio, Streamable HTTP or legacy HTTP+SSE, built on `rmcp`. Discovery is capability-gated and paginated, with repeated-cursor detection.
- `McpTransport` (`stdio`, `streamable_http`, `sse`, `with_cwd`, `with_env`, `with_inherited_env`, `with_header`, `with_request_timeout`, `with_connect_timeout`, `from_endpoint`); `Authorization: Bearer` headers are supported.
- Timeouts on every MCP transport: `McpTransport::with_request_timeout` (default `DEFAULT_REQUEST_TIMEOUT`, 60 s) bounds each request — every page of a list, `tools/call`, `resources/read`, `prompts/get` — and `McpTransport::with_connect_timeout` (default `DEFAULT_CONNECT_TIMEOUT`, 30 s) bounds spawning or opening the transport plus the `initialize` handshake. The legacy SSE transport's 30 s connect budget is now this connect timeout. A timed-out request is abandoned (rmcp sends `notifications/cancelled`) and the session stays usable.
- List requests stop with an error after 1000 pages, so a server that keeps sending new cursors cannot grow memory without bound.
- `McpResource` and `McpPrompt` implementing `Resource` and `Prompt`, with `McpResourceOutput` and `McpPromptOutput`.
- Orchestrator support: `orchestrator_schema`, `McpToolset`, `McpToolCall`, `McpOrchestratorOutput`, and `fetch_mcp_*` helpers.
- `AgentConfig::with_output_schema` to override the output JSON Schema sent to the model.
- Explicit `Error::Mcp` messages for unsupported `input_required` and task replies.
- URL userinfo, query and fragment are redacted from MCP error text; header values are marked sensitive. The Streamable HTTP transport uses rmcp's default reqwest client (same-origin 307/308 redirects only, at most 10) wrapped so that HTTP client errors carry no URL, which also keeps endpoint secrets out of rmcp's own `tracing` output.
- Legacy SSE hardening: same-origin POST endpoint, no redirects, connect bounded by the transport's connect timeout (30 s by default), 16 MiB per-event cap.
- The orchestrator schema does not require `serde_json/preserve_order`; `tool_name` is listed first in `required` only.
- The orchestrator schema strips each tool's own `$schema` keyword from its `anyOf` member. Gemini function declarations do not support `$ref`/`$defs`; schemas that use them may need flattening for Google models.
- TLS for the Streamable HTTP transport uses native-tls (rmcp's `reqwest-native-tls`), the same backend as this crate's reqwest 0.12 default. The `mcp` feature pulls in no rustls, aws-lc-rs or aws-lc-sys, so it cannot change which rustls crypto provider a downstream build selects.

### Known limitations

- A connected MCP server is inside the trust boundary: its tool names, descriptions and schemas reach the LLM unmarked; `McpToolset::dispatch` only calls discovered tools, and arguments are not validated against a tool's `input_schema`.
- MCP `input_required` (multi-round-trip) and task replies are reported as errors, not handled.

### Migration

```rust
// before
let tool = McpTool::new(def, endpoint, kind);

// after
let client = McpClient::connect_endpoint(endpoint, kind).await?;
let tools = client.tools().await?;
```
