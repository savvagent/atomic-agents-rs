//! Host-provided LLM and tool access for a WebAssembly guest.
//!
//! Inside a sandboxed guest there are no sockets: the host that runs the
//! component supplies the model and the tools. This module holds everything
//! the guest needs, in pure Rust with no dependency on the component
//! bindings, so it builds natively and for `wasm32-wasip2`:
//!
//! - [`LlmHost`] / [`McpHost`] - the two host capabilities, as traits. A
//!   component adapter implements them over its generated bindings; tests
//!   implement them with a mock.
//! - [`CompletionRequest`], [`Completion`], [`Message`], [`ToolSpec`] and the
//!   other data types - plain mirrors of the interface the host exposes.
//!   [`CompletionRequest`] has no field that selects or forces a tool; the
//!   host decides how to obtain structured output.
//! - [`HostLlmClient`] - an [`LlmClient`](crate::client::LlmClient) over an
//!   [`LlmHost`], so an `AtomicAgent` runs unchanged.
//! - [`HostTool`] / [`HostToolset`] - [`Tool`](crate::tool::Tool) adapters
//!   over an [`McpHost`].
//!
//! Host calls are synchronous. Run an agent with `run` when the `native`
//! feature is off, or drive `run_async` with any executor.

mod llm;
mod tool;

pub use llm::{
    Completion, CompletionRequest, ContentBlock, HostLlmClient, LlmError, LlmHost, Message, Role,
    ToolCall, ToolResult, ToolSpec, Usage,
};
pub use tool::{
    HostTool, HostToolInput, HostToolOutput, HostToolset, McpHost, ToolError, ToolInfo,
};
