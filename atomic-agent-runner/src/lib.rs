//! Sandboxed pipeline runner for `atomic-agents`.
//!
//! This crate holds the pure parts of the runner: the pipeline spec types,
//! templating, the condition language and static validation, plus the
//! [`interpreter`] that executes a pipeline over a [`host::Host`]. Nothing
//! here does I/O except through that trait.
//!
//! Built for `wasm32`, the crate is also a WebAssembly component of the
//! `runner` world in `wit/world.wit`: it implements [`host::Host`] over the
//! world's imports and exports `run` over [`interpreter::run`].

#[cfg(target_arch = "wasm32")]
mod component;
pub mod condition;
pub mod host;
pub mod interpreter;
pub mod schema;
pub mod spec;
pub mod template;
pub mod validate;
