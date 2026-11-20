//! The capabilities a pipeline run gets from its host.

use atomic_agents::guest::{LlmHost, McpHost};

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

/// Everything the interpreter may reach outside itself: the model and tools
/// (through the [`LlmHost`] and [`McpHost`] supertraits), a string key-value
/// state store, a log sink and a clock.
///
/// The interpreter has no other I/O, so a host decides exactly what a run can
/// touch. The component layer implements this over its imports; tests
/// implement it with a mock.
pub trait Host: LlmHost + McpHost {
    /// Reads a state value written by an earlier run, if any.
    fn state_get(&self, key: &str) -> Option<String>;
    /// Writes a state value for later runs.
    fn state_set(&self, key: &str, value: &str);
    /// Emits one log line.
    fn log(&self, level: Level, msg: &str);
    /// Milliseconds since the Unix epoch.
    fn now_ms(&self) -> u64;
}
