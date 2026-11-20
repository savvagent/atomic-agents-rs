//! Sandboxed pipeline runner for `atomic-agents`.
//!
//! This crate holds the pure parts of the runner: the pipeline spec types,
//! templating, the condition language and static validation, plus the
//! [`interpreter`] that executes a pipeline over a [`host::Host`]. Nothing
//! here does I/O except through that trait; the component bindings sit on
//! top of it.

pub mod condition;
pub mod host;
pub mod interpreter;
pub mod spec;
pub mod template;
pub mod validate;
