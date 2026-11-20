//! Component glue: the [`Host`] over the world's imports, and the `run`
//! export over [`interpreter::run`].
//!
//! The WIT types mirror `atomic_agents::guest` one to one, so everything here
//! is field-by-field conversion.

use ::atomic_agents::guest as g;

use crate::host::{CallOrigin, Host, Level};
use crate::interpreter::{self, RunError};

mod bindings {
    wit_bindgen::generate!({ world: "runner", path: "wit" });
}

use bindings::atomic_agents::runner::types as w;
use bindings::atomic_agents::runner::{clock, llm, log, mcp, state};

/// The [`Host`] backed by the component's imports. Zero-size: the imports
/// are free functions.
struct WitHost;

impl g::LlmHost for WitHost {
    fn complete(&self, req: g::CompletionRequest) -> Result<g::Completion, g::LlmError> {
        llm::complete(&req.into())
            .map(Into::into)
            .map_err(Into::into)
    }
}

impl Host for WitHost {
    fn list_tools(&self) -> Result<Vec<g::ToolInfo>, g::ToolError> {
        mcp::list_tools()
            .map(|tools| tools.into_iter().map(Into::into).collect())
            .map_err(Into::into)
    }

    // The 0.1.0 world's `call` has no origin parameter yet.
    fn call(&self, tool: &str, args: &str, _origin: CallOrigin) -> Result<String, g::ToolError> {
        mcp::call(tool, args).map_err(Into::into)
    }

    fn state_get(&self, key: &str) -> Option<String> {
        state::get(key)
    }

    fn state_set(&self, key: &str, value: &str) {
        state::set(key, value);
    }

    fn log(&self, level: Level, msg: &str) {
        log::emit(level.into(), msg);
    }

    fn now_ms(&self) -> u64 {
        clock::now()
    }
}

/// The exported component.
struct Component;

impl bindings::Guest for Component {
    fn run(spec: String, event: String) -> Result<String, w::RunError> {
        interpreter::run(&WitHost, &spec, &event).map_err(Into::into)
    }
}

bindings::export!(Component with_types_in bindings);

impl From<g::Role> for w::Role {
    fn from(r: g::Role) -> Self {
        match r {
            g::Role::User => Self::User,
            g::Role::Assistant => Self::Assistant,
        }
    }
}

impl From<g::ToolCall> for w::ToolCall {
    fn from(c: g::ToolCall) -> Self {
        Self {
            id: c.id,
            name: c.name,
            arguments: c.arguments,
        }
    }
}

impl From<w::ToolCall> for g::ToolCall {
    fn from(c: w::ToolCall) -> Self {
        Self {
            id: c.id,
            name: c.name,
            arguments: c.arguments,
        }
    }
}

impl From<g::ToolResult> for w::ToolResult {
    fn from(r: g::ToolResult) -> Self {
        Self {
            call_id: r.call_id,
            content: r.content,
            is_error: r.is_error,
        }
    }
}

impl From<g::ContentBlock> for w::ContentBlock {
    fn from(b: g::ContentBlock) -> Self {
        match b {
            g::ContentBlock::Text(t) => Self::Text(t),
            g::ContentBlock::ToolCall(c) => Self::ToolCall(c.into()),
            g::ContentBlock::ToolResult(r) => Self::ToolResult(r.into()),
        }
    }
}

impl From<g::Message> for w::Message {
    fn from(m: g::Message) -> Self {
        Self {
            role: m.role.into(),
            content: m.content.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<g::ToolSpec> for w::ToolSpec {
    fn from(s: g::ToolSpec) -> Self {
        Self {
            name: s.name,
            description: s.description,
            input_schema: s.input_schema,
        }
    }
}

impl From<g::CompletionRequest> for w::CompletionRequest {
    fn from(r: g::CompletionRequest) -> Self {
        Self {
            model: r.model,
            system: r.system,
            messages: r.messages.into_iter().map(Into::into).collect(),
            tools: r.tools.into_iter().map(Into::into).collect(),
            output_schema: r.output_schema,
            max_tokens: r.max_tokens,
            params: r.params,
        }
    }
}

impl From<w::Usage> for g::Usage {
    fn from(u: w::Usage) -> Self {
        Self {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
        }
    }
}

impl From<w::Completion> for g::Completion {
    fn from(c: w::Completion) -> Self {
        Self {
            text: c.text,
            tool_calls: c.tool_calls.into_iter().map(Into::into).collect(),
            stop_reason: c.stop_reason,
            usage: c.usage.map(Into::into),
        }
    }
}

impl From<w::LlmError> for g::LlmError {
    fn from(e: w::LlmError) -> Self {
        match e {
            w::LlmError::Refused(m) => Self::Refused(m),
            w::LlmError::LimitExceeded(m) => Self::LimitExceeded(m),
            w::LlmError::InvalidRequest(m) => Self::InvalidRequest(m),
            w::LlmError::Unavailable(m) => Self::Unavailable(m),
        }
    }
}

impl From<g::LlmError> for w::LlmError {
    fn from(e: g::LlmError) -> Self {
        match e {
            g::LlmError::Refused(m) => Self::Refused(m),
            g::LlmError::LimitExceeded(m) => Self::LimitExceeded(m),
            g::LlmError::InvalidRequest(m) => Self::InvalidRequest(m),
            g::LlmError::Unavailable(m) => Self::Unavailable(m),
        }
    }
}

impl From<w::ToolInfo> for g::ToolInfo {
    fn from(t: w::ToolInfo) -> Self {
        Self {
            name: t.name,
            description: t.description,
            input_schema: t.input_schema,
        }
    }
}

impl From<w::ToolError> for g::ToolError {
    fn from(e: w::ToolError) -> Self {
        match e {
            w::ToolError::Denied(m) => Self::Denied(m),
            w::ToolError::NotFound(m) => Self::NotFound(m),
            w::ToolError::InvalidArguments(m) => Self::InvalidArguments(m),
            w::ToolError::Failed(m) => Self::Failed(m),
            w::ToolError::LimitExceeded(m) => Self::LimitExceeded(m),
        }
    }
}

impl From<g::ToolError> for w::ToolError {
    fn from(e: g::ToolError) -> Self {
        match e {
            g::ToolError::Denied(m) => Self::Denied(m),
            g::ToolError::NotFound(m) => Self::NotFound(m),
            g::ToolError::InvalidArguments(m) => Self::InvalidArguments(m),
            g::ToolError::Failed(m) => Self::Failed(m),
            g::ToolError::LimitExceeded(m) => Self::LimitExceeded(m),
        }
    }
}

impl From<Level> for w::Level {
    fn from(l: Level) -> Self {
        match l {
            Level::Trace => Self::Trace,
            Level::Debug => Self::Debug,
            Level::Info => Self::Info,
            Level::Warn => Self::Warn,
            Level::Error => Self::Error,
        }
    }
}

impl From<RunError> for w::RunError {
    fn from(e: RunError) -> Self {
        match e {
            RunError::InvalidSpec(m) => Self::InvalidSpec(m),
            RunError::InvalidEvent(m) => Self::InvalidEvent(m),
            RunError::LimitExceeded(m) => Self::LimitExceeded(m),
            RunError::Llm(e) => Self::Llm(e.into()),
            RunError::Tool(e) => Self::Tool(e.into()),
            RunError::StepFailed { step, message } => {
                Self::StepFailed(w::StepFailure { step, message })
            }
        }
    }
}
