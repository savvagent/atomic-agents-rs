//! Sandboxed pipeline runner for `atomic-agents`.
//!
//! This crate holds the pure parts of the runner: the pipeline spec types,
//! templating, the condition language and static validation. None of it does
//! I/O; the interpreter and the component bindings sit on top of it.

pub mod condition;
pub mod spec;
pub mod template;
pub mod validate;
