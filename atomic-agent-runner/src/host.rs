//! The capabilities a pipeline run gets from its host.

use atomic_agents::guest::{LlmHost, ToolError, ToolInfo};

/// Severity of a [`Host::log`] line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// Very fine detail.
    Trace,
    /// Diagnostic detail.
    Debug,
    /// Normal progress.
    Info,
    /// Something unexpected that did not stop the run.
    Warn,
    /// A failure.
    Error,
}

/// Which kind of step made a [`Host::call`].
///
/// The interpreter sets it from the code path making the call, never from
/// model output or from the call's arguments. [`CallOrigin::ActStep`] means
/// "some act step of this spec named this tool"; it carries no step id.
///
/// A host should treat the origin as one check among several, not as proof:
/// for example, refuse a write tool unless the origin is `ActStep` and an act
/// step of the deployed spec names that tool, and still check the arguments,
/// since act arguments can be rendered from earlier model output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallOrigin {
    /// A tool call the model asked for in an agent step.
    Model,
    /// A `tool` step.
    ToolStep,
    /// An `act` step.
    ActStep,
}

/// Everything the interpreter may reach outside itself: the model (through
/// the [`LlmHost`] supertrait), the host's tools, a string key-value state
/// store, a log sink and a clock.
///
/// The interpreter has no other I/O, so a host decides exactly what a run can
/// touch. The component layer implements this over its imports; tests
/// implement it with a mock.
pub trait Host: LlmHost {
    /// Lists the tools the model may be offered in this run.
    ///
    /// # Errors
    ///
    /// The host's [`ToolError`] if it cannot list its tools.
    fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolError>;
    /// Calls `tool` with `args`, a JSON document, and returns its output.
    /// `origin` is the kind of step making the call.
    ///
    /// # Errors
    ///
    /// The host's [`ToolError`] if it refuses or fails the call.
    fn call(&self, tool: &str, args: &str, origin: CallOrigin) -> Result<String, ToolError>;
    /// Reads a state value written by an earlier run, if any.
    fn state_get(&self, key: &str) -> Option<String>;
    /// Writes a state value for later runs. The interpreter calls this only
    /// after a run has succeeded, once per key written during it.
    fn state_set(&self, key: &str, value: &str);
    /// Emits one log line.
    fn log(&self, level: Level, msg: &str);
    /// Milliseconds since the Unix epoch.
    fn now_ms(&self) -> u64;
}
