use serde_json::Value;

use crate::error::{Error, Result};
use crate::pipeline::MergeStrategy;

/// Merges multiple input values according to the given strategy.
pub fn merge_values(strategy: &MergeStrategy, inputs: Vec<Value>) -> Result<Value> {
    match strategy {
        MergeStrategy::WaitAll => Ok(Value::Array(inputs)),
        MergeStrategy::FirstReady => inputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::Pipeline("No inputs to merge".to_string())),
        MergeStrategy::MergeObjects => {
            let mut merged = serde_json::Map::new();
            for input in inputs {
                match input {
                    Value::Object(map) => {
                        deep_merge(&mut merged, map);
                    }
                    other => {
                        return Err(Error::Pipeline(format!(
                            "MergeObjects requires object inputs, got: {}",
                            value_type_name(&other)
                        )));
                    }
                }
            }
            Ok(Value::Object(merged))
        }
    }
}

/// Deep-merges `source` into `target`. Source values overwrite target values
/// for scalar types; objects are recursively merged.
fn deep_merge(target: &mut serde_json::Map<String, Value>, source: serde_json::Map<String, Value>) {
    for (key, value) in source {
        match (target.get_mut(&key), &value) {
            (Some(Value::Object(existing)), Value::Object(incoming)) => {
                deep_merge(existing, incoming.clone());
            }
            _ => {
                target.insert(key, value);
            }
        }
    }
}

fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_wait_all() {
        let inputs = vec![json!({"a": 1}), json!({"b": 2})];
        let result = merge_values(&MergeStrategy::WaitAll, inputs).unwrap();
        assert_eq!(result, json!([{"a": 1}, {"b": 2}]));
    }

    #[test]
    fn test_first_ready() {
        let inputs = vec![json!({"a": 1}), json!({"b": 2})];
        let result = merge_values(&MergeStrategy::FirstReady, inputs).unwrap();
        assert_eq!(result, json!({"a": 1}));
    }

    #[test]
    fn test_first_ready_empty() {
        let result = merge_values(&MergeStrategy::FirstReady, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_objects() {
        let inputs = vec![
            json!({"name": "Alice", "meta": {"role": "admin"}}),
            json!({"email": "alice@test.com", "meta": {"active": true}}),
        ];
        let result = merge_values(&MergeStrategy::MergeObjects, inputs).unwrap();
        assert_eq!(
            result,
            json!({"name": "Alice", "email": "alice@test.com", "meta": {"role": "admin", "active": true}})
        );
    }

    #[test]
    fn test_merge_objects_non_object_error() {
        let inputs = vec![json!("not an object")];
        let result = merge_values(&MergeStrategy::MergeObjects, inputs);
        assert!(result.is_err());
    }
}
