//! `{{ path }}` templating over JSON values.

use serde_json::Value;

/// Templating failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TemplateError {
    /// A `{{ path }}` named nothing in the context.
    #[error("unresolved {{{{{0}}}}}")]
    Unresolved(String),
}

/// Looks up a dotted path; segments index objects by key and arrays by
/// decimal index. A present `null` is returned as `Some(&Value::Null)`.
///
/// ```
/// use atomic_agent_runner::template::lookup;
/// let ctx = serde_json::json!({"a": {"b": [10, 20]}});
/// assert_eq!(lookup(&ctx, "a.b.1"), Some(&serde_json::json!(20)));
/// assert_eq!(lookup(&ctx, "a.c"), None);
/// ```
pub fn lookup<'a>(ctx: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }
    let mut cur = ctx;
    for seg in path.split('.') {
        cur = match cur {
            Value::Object(m) => m.get(seg)?,
            Value::Array(a) => a.get(seg.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

/// Renders every string inside `template` against `ctx`.
///
/// A string that is exactly `{{ path }}` becomes the value at `path`; a string
/// with embedded `{{path}}` is interpolated (strings raw, other values as
/// JSON); an unclosed `{{` stays literal.
///
/// ```
/// use atomic_agent_runner::template::render;
/// use serde_json::json;
/// let ctx = json!({"n": 3, "s": "x"});
/// assert_eq!(render(&json!("{{ n }}"), &ctx).unwrap(), json!(3));
/// assert_eq!(render(&json!("a {{s}} {{n}}"), &ctx).unwrap(), json!("a x 3"));
/// ```
pub fn render(template: &Value, ctx: &Value) -> Result<Value, TemplateError> {
    match template {
        Value::String(s) => render_str(s, ctx),
        Value::Array(a) => a
            .iter()
            .map(|v| render(v, ctx))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| render(v, ctx).map(|v| (k.clone(), v)))
            .collect::<Result<serde_json::Map<_, _>, _>>()
            .map(Value::Object),
        other => Ok(other.clone()),
    }
}

fn resolve<'a>(ctx: &'a Value, raw: &str) -> Result<&'a Value, TemplateError> {
    let path = raw.trim();
    lookup(ctx, path).ok_or_else(|| TemplateError::Unresolved(path.to_string()))
}

fn render_str(s: &str, ctx: &Value) -> Result<Value, TemplateError> {
    if let Some(inner) = s.strip_prefix("{{").and_then(|r| r.strip_suffix("}}"))
        && !inner.contains("{{")
        && !inner.contains("}}")
    {
        return resolve(ctx, inner).cloned();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("{{") {
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else { break };
        out.push_str(&rest[..start]);
        match resolve(ctx, &after[..end])? {
            Value::String(v) => out.push_str(v),
            other => out.push_str(&other.to_string()),
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(Value::String(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> Value {
        json!({"event": {"n": 3, "name": "Ann", "tags": ["a", "b"], "nil": null}, "x": {"y": {"z": true}}})
    }

    #[test]
    fn whole_value_keeps_type() {
        assert_eq!(render(&json!("{{ event.n }}"), &ctx()).unwrap(), json!(3));
        assert_eq!(
            render(&json!("{{event.tags}}"), &ctx()).unwrap(),
            json!(["a", "b"])
        );
        assert_eq!(
            render(&json!("{{ event.nil }}"), &ctx()).unwrap(),
            Value::Null
        );
        assert_eq!(render(&json!("{{ x.y.z }}"), &ctx()).unwrap(), json!(true));
    }

    #[test]
    fn embedded_interpolates() {
        assert_eq!(
            render(
                &json!("hi {{event.name}}, n={{ event.n }}, t={{event.tags}}"),
                &ctx()
            )
            .unwrap(),
            json!("hi Ann, n=3, t=[\"a\",\"b\"]")
        );
        assert_eq!(
            render(&json!("v={{event.nil}}"), &ctx()).unwrap(),
            json!("v=null")
        );
    }

    #[test]
    fn missing_path_errors() {
        assert_eq!(
            render(&json!("{{ event.zzz }}"), &ctx()),
            Err(TemplateError::Unresolved("event.zzz".into()))
        );
        assert_eq!(
            render(&json!("a {{ nope }} b"), &ctx()),
            Err(TemplateError::Unresolved("nope".into()))
        );
        assert!(render(&json!("{{ }}"), &ctx()).is_err());
    }

    #[test]
    fn nested_structures_recurse() {
        let t = json!({"a": ["{{ event.n }}", {"b": "n{{event.n}}"}], "c": 1, "d": null});
        assert_eq!(
            render(&t, &ctx()).unwrap(),
            json!({"a": [3, {"b": "n3"}], "c": 1, "d": null})
        );
    }

    #[test]
    fn unclosed_is_literal() {
        assert_eq!(render(&json!("a {{ b"), &ctx()).unwrap(), json!("a {{ b"));
        assert_eq!(
            render(&json!("{{event.n}} then {{ open"), &ctx()).unwrap(),
            json!("3 then {{ open")
        );
    }

    #[test]
    fn lookup_indexes_arrays() {
        assert_eq!(lookup(&ctx(), "event.tags.1"), Some(&json!("b")));
        assert_eq!(lookup(&ctx(), "event.tags.9"), None);
        assert_eq!(lookup(&ctx(), "event.tags.x"), None);
        assert_eq!(lookup(&ctx(), "event.n.deeper"), None);
        assert_eq!(lookup(&ctx(), ""), None);
    }
}
