//! JSON Schema compilation for agent output schemas.
//!
//! A spec's `output_schema` is untrusted, and `jsonschema` resolves an
//! external `$ref` through its retriever while compiling. Its default
//! retriever fetches `http(s)` and reads `file` URIs whenever the
//! `resolve-http` / `resolve-file` features are on, which another crate in
//! the same build can turn on through feature unification. Every validator
//! in this crate is therefore built by [`compile`], whose retriever refuses
//! everything: a schema can reference only itself (`#...`) and the bundled
//! meta-schemas.

use jsonschema::{Retrieve, Uri, Validator};
use serde_json::Value;

/// A retriever that never fetches anything.
struct NoRetrieval;

impl Retrieve for NoRetrieval {
    fn retrieve(
        &self,
        _uri: &Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("external schema retrieval is disabled".into())
    }
}

/// Compiles `schema` into a validator without retrieving any external
/// resource; a `$ref` to one is a compile error.
///
/// ```
/// use atomic_agent_runner::schema::compile;
/// use serde_json::json;
/// assert!(compile(&json!({"type": "string"})).unwrap().is_valid(&json!("x")));
/// let err = compile(&json!({"$ref": "https://example.com/s.json"})).unwrap_err();
/// assert!(err.contains("retrieval is disabled"), "{err}");
/// ```
pub fn compile(schema: &Value) -> Result<Validator, String> {
    jsonschema::options()
        .with_retriever(NoRetrieval)
        .build(schema)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::net::TcpListener;

    #[test]
    fn local_schemas_compile() {
        let v = compile(&json!({"type": "object", "required": ["a"],
            "$defs": {"n": {"type": "number"}},
            "properties": {"a": {"$ref": "#/$defs/n"}}}))
        .unwrap();
        assert!(v.is_valid(&json!({"a": 1})));
        assert!(!v.is_valid(&json!({"a": "x"})));
    }

    #[test]
    fn remote_refs_are_refused_without_a_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let err = compile(&json!({"$ref": format!("http://127.0.0.1:{port}/x")})).unwrap_err();
        assert!(err.contains("retrieval is disabled"), "{err}");
        assert!(
            matches!(listener.accept(), Err(e) if e.kind() == std::io::ErrorKind::WouldBlock),
            "the schema compiler connected to the listener"
        );
    }

    #[test]
    fn file_refs_are_refused() {
        let err = compile(&json!({"$ref": "file:///etc/hostname"})).unwrap_err();
        assert!(err.contains("retrieval is disabled"), "{err}");
    }
}
