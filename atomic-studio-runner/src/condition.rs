use serde_json::Value;

use crate::error::{Error, Result};
use crate::pipeline::ConditionExpr;

/// Evaluates a condition expression against a JSON value.
pub fn evaluate(expr: &ConditionExpr, data: &Value) -> Result<bool> {
    match expr {
        ConditionExpr::Equals { field, value } => {
            let field_val = get_field(data, field)?;
            Ok(*field_val == *value)
        }
        ConditionExpr::Contains { field, value } => {
            let field_val = get_field(data, field)?;
            match &field_val {
                Value::String(s) => Ok(s.contains(value.as_str())),
                Value::Array(arr) => Ok(arr.iter().any(|v| {
                    v.as_str().is_some_and(|s| s == value)
                })),
                _ => Ok(false),
            }
        }
        ConditionExpr::GreaterThan { field, value } => {
            let field_val = get_field(data, field)?;
            Ok(as_f64(&field_val).is_some_and(|n| n > *value))
        }
        ConditionExpr::LessThan { field, value } => {
            let field_val = get_field(data, field)?;
            Ok(as_f64(&field_val).is_some_and(|n| n < *value))
        }
        ConditionExpr::Matches { field, pattern } => {
            let field_val = get_field(data, field)?;
            match field_val.as_str() {
                Some(s) => {
                    let re = regex_lite::Regex::new(pattern)
                        .map_err(|e| Error::Condition(format!("Invalid regex '{pattern}': {e}")))?;
                    Ok(re.is_match(s))
                }
                None => Ok(false),
            }
        }
        ConditionExpr::Exists { field } => Ok(get_field(data, field).is_ok()),
        ConditionExpr::And { conditions } => {
            for cond in conditions {
                if !evaluate(cond, data)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        ConditionExpr::Or { conditions } => {
            for cond in conditions {
                if evaluate(cond, data)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        ConditionExpr::Not { condition } => Ok(!evaluate(condition, data)?),
    }
}

/// Extracts a field value from a JSON object using dot notation.
/// e.g., "foo.bar.baz" navigates into nested objects.
fn get_field<'a>(data: &'a Value, field: &str) -> Result<&'a Value> {
    let mut current = data;
    for part in field.split('.') {
        current = current.get(part).ok_or_else(|| {
            Error::Condition(format!("Field '{field}' not found in data"))
        })?;
    }
    if current.is_null() {
        return Err(Error::Condition(format!("Field '{field}' is null")));
    }
    Ok(current)
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_equals() {
        let data = json!({"status": "active", "count": 5});
        let expr = ConditionExpr::Equals {
            field: "status".into(),
            value: json!("active"),
        };
        assert!(evaluate(&expr, &data).unwrap());

        let expr = ConditionExpr::Equals {
            field: "count".into(),
            value: json!(5),
        };
        assert!(evaluate(&expr, &data).unwrap());

        let expr = ConditionExpr::Equals {
            field: "status".into(),
            value: json!("inactive"),
        };
        assert!(!evaluate(&expr, &data).unwrap());
    }

    #[test]
    fn test_contains() {
        let data = json!({"message": "hello world"});
        let expr = ConditionExpr::Contains {
            field: "message".into(),
            value: "world".into(),
        };
        assert!(evaluate(&expr, &data).unwrap());
    }

    #[test]
    fn test_greater_than() {
        let data = json!({"score": 85});
        let expr = ConditionExpr::GreaterThan {
            field: "score".into(),
            value: 80.0,
        };
        assert!(evaluate(&expr, &data).unwrap());

        let expr = ConditionExpr::GreaterThan {
            field: "score".into(),
            value: 90.0,
        };
        assert!(!evaluate(&expr, &data).unwrap());
    }

    #[test]
    fn test_exists() {
        let data = json!({"name": "test"});
        assert!(evaluate(&ConditionExpr::Exists { field: "name".into() }, &data).unwrap());
        assert!(!evaluate(&ConditionExpr::Exists { field: "missing".into() }, &data).unwrap());
    }

    #[test]
    fn test_nested_field() {
        let data = json!({"user": {"role": "admin"}});
        let expr = ConditionExpr::Equals {
            field: "user.role".into(),
            value: json!("admin"),
        };
        assert!(evaluate(&expr, &data).unwrap());
    }

    #[test]
    fn test_and_or_not() {
        let data = json!({"a": 1, "b": 2});
        let expr = ConditionExpr::And {
            conditions: vec![
                ConditionExpr::Equals { field: "a".into(), value: json!(1) },
                ConditionExpr::Equals { field: "b".into(), value: json!(2) },
            ],
        };
        assert!(evaluate(&expr, &data).unwrap());

        let expr = ConditionExpr::Not {
            condition: Box::new(ConditionExpr::Equals {
                field: "a".into(),
                value: json!(99),
            }),
        };
        assert!(evaluate(&expr, &data).unwrap());
    }
}
