//! The condition language.
//!
//! [`ConditionExpr`] and [`evaluate`] port `atomic-studio-runner`'s
//! `condition.rs` (same serde tags, same semantics): a leaf that reads a
//! missing or null field is an error that propagates through `!`, `&&`, `||`;
//! only `exists` inspects a missing field without erroring. [`parse`] turns
//! the string form into the same tree.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::spec::Condition;
use crate::template::lookup;

/// A condition expression, evaluated against a JSON value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op")]
#[serde(rename_all = "snake_case")]
pub enum ConditionExpr {
    /// Field equals a value.
    Equals { field: String, value: Value },
    /// Field does not equal a value (errors on a missing or null field).
    NotEquals { field: String, value: Value },
    /// Field (string or array of strings) contains a string.
    Contains { field: String, value: String },
    /// Field is greater than a number.
    GreaterThan { field: String, value: f64 },
    /// Field is at least a number.
    GreaterOrEqual { field: String, value: f64 },
    /// Field is less than a number.
    LessThan { field: String, value: f64 },
    /// Field is at most a number.
    LessOrEqual { field: String, value: f64 },
    /// Field matches a regex.
    Matches { field: String, pattern: String },
    /// Field equals one of the values.
    In { field: String, values: Vec<Value> },
    /// Field exists and is not null.
    Exists { field: String },
    /// All sub-conditions hold.
    And { conditions: Vec<ConditionExpr> },
    /// At least one sub-condition holds.
    Or { conditions: Vec<ConditionExpr> },
    /// Negates a sub-condition.
    Not { condition: Box<ConditionExpr> },
}

/// Condition failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConditionError {
    /// A leaf read a missing or null field.
    #[error("{0}")]
    Field(String),
    /// A `matches` pattern did not compile.
    #[error("invalid regex '{pattern}': {message}")]
    Regex { pattern: String, message: String },
    /// The expression string did not parse.
    #[error("parse error: {0}")]
    Parse(String),
}

/// Whether a `when` guard lets its step run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhenOutcome {
    /// Run the step.
    True,
    /// Skip the step.
    False,
    /// A field was missing or null: skip the step and record this reason.
    FieldError(String),
}

/// Evaluates `expr` against `data`.
///
/// ```
/// use atomic_agent_runner::condition::{evaluate, parse};
/// let expr = parse("a.n >= 2 && a.s in ['x', 'y']").unwrap();
/// let data = serde_json::json!({"a": {"n": 2, "s": "y"}});
/// assert_eq!(evaluate(&expr, &data), Ok(true));
/// ```
pub fn evaluate(expr: &ConditionExpr, data: &Value) -> Result<bool, ConditionError> {
    use ConditionExpr as C;
    match expr {
        C::Equals { field, value } => Ok(get_field(data, field)? == value),
        C::NotEquals { field, value } => Ok(get_field(data, field)? != value),
        C::Contains { field, value } => Ok(match get_field(data, field)? {
            Value::String(s) => s.contains(value.as_str()),
            Value::Array(a) => a.iter().any(|v| v.as_str() == Some(value.as_str())),
            _ => false,
        }),
        C::GreaterThan { field, value } => {
            Ok(get_field(data, field)?.as_f64().is_some_and(|n| n > *value))
        }
        C::GreaterOrEqual { field, value } => Ok(get_field(data, field)?
            .as_f64()
            .is_some_and(|n| n >= *value)),
        C::LessThan { field, value } => {
            Ok(get_field(data, field)?.as_f64().is_some_and(|n| n < *value))
        }
        C::LessOrEqual { field, value } => Ok(get_field(data, field)?
            .as_f64()
            .is_some_and(|n| n <= *value)),
        C::Matches { field, pattern } => match get_field(data, field)?.as_str() {
            Some(s) => Ok(compile_regex(pattern)?.is_match(s)),
            None => Ok(false),
        },
        C::In { field, values } => Ok(values.contains(get_field(data, field)?)),
        C::Exists { field } => Ok(get_field(data, field).is_ok()),
        C::And { conditions } => {
            for c in conditions {
                if !evaluate(c, data)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        C::Or { conditions } => {
            for c in conditions {
                if evaluate(c, data)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        C::Not { condition } => Ok(!evaluate(condition, data)?),
    }
}

/// Reads a dotted field; missing and null are both errors. Unlike the studio
/// port source, numeric segments also index arrays (a superset).
fn get_field<'a>(data: &'a Value, field: &str) -> Result<&'a Value, ConditionError> {
    match lookup(data, field) {
        None => Err(ConditionError::Field(format!(
            "field '{field}' not found in data"
        ))),
        Some(Value::Null) => Err(ConditionError::Field(format!("field '{field}' is null"))),
        Some(v) => Ok(v),
    }
}

/// Deepest allowed condition tree (a leaf alone has depth 1).
pub const MAX_COND_DEPTH: usize = 32;
/// Most nodes (leaves and combinators) in one condition tree.
pub const MAX_COND_NODES: usize = 256;
/// Most values in one `in` list.
pub const MAX_IN_VALUES: usize = 100;
/// Longest `matches` pattern, in bytes.
pub const MAX_PATTERN_LEN: usize = 256;
/// Compiled-regex size cap, so a hostile pattern cannot blow up memory.
const REGEX_SIZE_LIMIT: usize = 1 << 20;

fn compile_regex(pattern: &str) -> Result<regex_lite::Regex, ConditionError> {
    regex_lite::RegexBuilder::new(pattern)
        .size_limit(REGEX_SIZE_LIMIT)
        .build()
        .map_err(|e| ConditionError::Regex {
            pattern: pattern.to_string(),
            message: e.to_string(),
        })
}

/// Rejects trees that are too deep or large, `in` lists that are too long,
/// and `matches` patterns that are too long or do not compile within the
/// size limit. Applies to structured conditions and to [`parse`] output.
pub fn check_bounds(expr: &ConditionExpr) -> Result<(), ConditionError> {
    let mut nodes = 0;
    bounds(expr, 1, &mut nodes)
}

fn bounds(expr: &ConditionExpr, depth: usize, nodes: &mut usize) -> Result<(), ConditionError> {
    if depth > MAX_COND_DEPTH {
        return perr(format!("condition nested deeper than {MAX_COND_DEPTH}"));
    }
    *nodes += 1;
    if *nodes > MAX_COND_NODES {
        return perr(format!("condition has more than {MAX_COND_NODES} nodes"));
    }
    match expr {
        ConditionExpr::In { values, .. } if values.len() > MAX_IN_VALUES => {
            perr(format!("'in' list has more than {MAX_IN_VALUES} values"))
        }
        ConditionExpr::Matches { pattern, .. } => {
            if pattern.len() > MAX_PATTERN_LEN {
                return perr(format!("pattern longer than {MAX_PATTERN_LEN} bytes"));
            }
            compile_regex(pattern).map(drop)
        }
        ConditionExpr::And { conditions } | ConditionExpr::Or { conditions } => conditions
            .iter()
            .try_for_each(|c| bounds(c, depth + 1, nodes)),
        ConditionExpr::Not { condition } => bounds(condition, depth + 1, nodes),
        _ => Ok(()),
    }
}

/// Evaluates a step's `when` guard. A missing or null field yields
/// [`WhenOutcome::FieldError`]; a string that does not parse, or an invalid
/// regex, is an `Err` (validation reports parse errors before a run).
pub fn when_holds(cond: &Condition, ctx: &Value) -> Result<WhenOutcome, ConditionError> {
    let result = match cond {
        Condition::Expr(src) => evaluate(&parse(src)?, ctx),
        Condition::Structured(expr) => evaluate(expr, ctx),
    };
    match result {
        Ok(true) => Ok(WhenOutcome::True),
        Ok(false) => Ok(WhenOutcome::False),
        Err(ConditionError::Field(msg)) => Ok(WhenOutcome::FieldError(msg)),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------- parser

const MAX_DEPTH: usize = 32;
const MAX_SOURCE_LEN: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Path(String),
    Lit(Value),
    In,
    AndAnd,
    OrOr,
    Bang,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
}

fn perr<T>(msg: impl Into<String>) -> Result<T, ConditionError> {
    Err(ConditionError::Parse(msg.into()))
}

fn lex(src: &str) -> Result<Vec<Tok>, ConditionError> {
    if src.len() > MAX_SOURCE_LEN {
        return perr("expression too long");
    }
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            c if c.is_whitespace() => i += 1,
            '&' if next == Some('&') => {
                toks.push(Tok::AndAnd);
                i += 2;
            }
            '|' if next == Some('|') => {
                toks.push(Tok::OrOr);
                i += 2;
            }
            '=' if next == Some('=') => {
                toks.push(Tok::Eq);
                i += 2;
            }
            '!' if next == Some('=') => {
                toks.push(Tok::Ne);
                i += 2;
            }
            '!' => {
                toks.push(Tok::Bang);
                i += 1;
            }
            '<' | '>' => {
                let or_eq = next == Some('=');
                toks.push(match (c, or_eq) {
                    ('<', false) => Tok::Lt,
                    ('<', true) => Tok::Le,
                    (_, false) => Tok::Gt,
                    (_, true) => Tok::Ge,
                });
                i += if or_eq { 2 } else { 1 };
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            '[' => {
                toks.push(Tok::LBracket);
                i += 1;
            }
            ']' => {
                toks.push(Tok::RBracket);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            '\'' | '"' => {
                let (s, end) = lex_string(&chars, i)?;
                toks.push(Tok::Lit(Value::String(s)));
                i = end;
            }
            c if c.is_ascii_digit() || (c == '-' && next.is_some_and(|n| n.is_ascii_digit())) => {
                let start = i;
                i += 1;
                while i < chars.len() && "0123456789.eE+-".contains(chars[i]) {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                toks.push(Tok::Lit(number(&text)?));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_alphanumeric() || matches!(chars[i], '_' | '.' | '-'))
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                toks.push(match word.as_str() {
                    "true" => Tok::Lit(Value::Bool(true)),
                    "false" => Tok::Lit(Value::Bool(false)),
                    "null" => Tok::Lit(Value::Null),
                    "in" => Tok::In,
                    _ => {
                        if word.split('.').any(str::is_empty) {
                            return perr(format!("malformed path '{word}'"));
                        }
                        Tok::Path(word)
                    }
                });
            }
            other => return perr(format!("unexpected character '{other}'")),
        }
    }
    Ok(toks)
}

fn number(text: &str) -> Result<Value, ConditionError> {
    if let Ok(i) = text.parse::<i64>() {
        return Ok(Value::from(i));
    }
    match text
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
    {
        Some(n) => Ok(Value::Number(n)),
        None => perr(format!("invalid number '{text}'")),
    }
}

/// Reads a quoted string starting at `start`; returns it and the index after
/// the closing quote. Escapes: `\\`, `\'`, `\"`, `\n`, `\t`, `\r`.
fn lex_string(chars: &[char], start: usize) -> Result<(String, usize), ConditionError> {
    let quote = chars[start];
    let mut out = String::new();
    let mut i = start + 1;
    while i < chars.len() {
        match chars[i] {
            c if c == quote => return Ok((out, i + 1)),
            '\\' => {
                i += 1;
                match chars.get(i) {
                    Some('\\') => out.push('\\'),
                    Some('\'') => out.push('\''),
                    Some('"') => out.push('"'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some(c) => return perr(format!("unknown escape '\\{c}'")),
                    None => break,
                }
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    perr("unterminated string")
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn or(&mut self, depth: usize) -> Result<ConditionExpr, ConditionError> {
        let mut parts = vec![self.and(depth)?];
        while self.eat(&Tok::OrOr) {
            parts.push(self.and(depth)?);
        }
        Ok(one_or(parts, |conditions| ConditionExpr::Or { conditions }))
    }

    fn and(&mut self, depth: usize) -> Result<ConditionExpr, ConditionError> {
        let mut parts = vec![self.unary(depth)?];
        while self.eat(&Tok::AndAnd) {
            parts.push(self.unary(depth)?);
        }
        Ok(one_or(parts, |conditions| ConditionExpr::And {
            conditions,
        }))
    }

    fn unary(&mut self, depth: usize) -> Result<ConditionExpr, ConditionError> {
        if depth > MAX_DEPTH {
            return perr("expression nested too deeply");
        }
        if self.eat(&Tok::Bang) {
            let inner = self.unary(depth + 1)?;
            return Ok(ConditionExpr::Not {
                condition: Box::new(inner),
            });
        }
        if self.eat(&Tok::LParen) {
            let inner = self.or(depth + 1)?;
            if !self.eat(&Tok::RParen) {
                return perr("expected ')'");
            }
            return Ok(inner);
        }
        self.cmp()
    }

    fn cmp(&mut self) -> Result<ConditionExpr, ConditionError> {
        let field = match self.bump() {
            Some(Tok::Path(p)) => p,
            Some(Tok::Lit(_)) => {
                return perr("a literal on the left of a comparison is not supported");
            }
            Some(t) => return perr(format!("expected a field path, found {t:?}")),
            None => return perr("unexpected end of expression"),
        };
        let op = match self.peek() {
            Some(t @ (Tok::Eq | Tok::Ne | Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge | Tok::In)) => {
                t.clone()
            }
            _ => return Ok(ConditionExpr::Exists { field }),
        };
        self.pos += 1;
        if op == Tok::In {
            return self.in_list(field);
        }
        let value = self.literal()?;
        Ok(match op {
            Tok::Eq => ConditionExpr::Equals { field, value },
            Tok::Ne => ConditionExpr::NotEquals { field, value },
            _ => {
                let Some(value) = value.as_f64() else {
                    return perr("ordering comparisons need a number");
                };
                match op {
                    Tok::Lt => ConditionExpr::LessThan { field, value },
                    Tok::Le => ConditionExpr::LessOrEqual { field, value },
                    Tok::Gt => ConditionExpr::GreaterThan { field, value },
                    _ => ConditionExpr::GreaterOrEqual { field, value },
                }
            }
        })
    }

    fn in_list(&mut self, field: String) -> Result<ConditionExpr, ConditionError> {
        if !self.eat(&Tok::LBracket) {
            return perr("expected '[' after 'in'");
        }
        let mut values = vec![self.literal()?];
        while self.eat(&Tok::Comma) {
            values.push(self.literal()?);
        }
        if !self.eat(&Tok::RBracket) {
            return perr("expected ']'");
        }
        Ok(ConditionExpr::In { field, values })
    }

    fn literal(&mut self) -> Result<Value, ConditionError> {
        match self.bump() {
            Some(Tok::Lit(v)) => Ok(v),
            Some(t) => perr(format!("expected a literal, found {t:?}")),
            None => perr("expected a literal, found end of expression"),
        }
    }
}

fn one_or(
    parts: Vec<ConditionExpr>,
    wrap: impl FnOnce(Vec<ConditionExpr>) -> ConditionExpr,
) -> ConditionExpr {
    match <[ConditionExpr; 1]>::try_from(parts) {
        Ok([only]) => only,
        Err(parts) => wrap(parts),
    }
}

/// Parses an expression string.
///
/// Grammar: `or := and ('||' and)*; and := unary ('&&' unary)*;
/// unary := '!' unary | '(' or ')' | cmp; cmp := path (op literal |
/// 'in' '[' literal (',' literal)* ']')?`. A bare path means `exists`.
///
/// ```
/// use atomic_agent_runner::condition::{parse, ConditionExpr};
/// assert!(matches!(parse("!a").unwrap(), ConditionExpr::Not { .. }));
/// assert!(parse("1 == a").is_err());
/// ```
pub fn parse(src: &str) -> Result<ConditionExpr, ConditionError> {
    let mut p = Parser {
        toks: lex(src)?,
        pos: 0,
    };
    let expr = p.or(0)?;
    if let Some(t) = p.peek() {
        return perr(format!("unexpected trailing {t:?}"));
    }
    check_bounds(&expr)?;
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ev(src: &str, data: &Value) -> Result<bool, ConditionError> {
        evaluate(&parse(src).unwrap(), data)
    }

    // Parity with the studio runner's tests.
    #[test]
    fn studio_parity() {
        let data = json!({"status": "active", "count": 5, "message": "hello world",
            "score": 85, "user": {"role": "admin"}, "a": 1, "b": 2, "name": "test"});
        let eq = |f: &str, v: Value| ConditionExpr::Equals {
            field: f.into(),
            value: v,
        };
        assert_eq!(evaluate(&eq("status", json!("active")), &data), Ok(true));
        assert_eq!(evaluate(&eq("count", json!(5)), &data), Ok(true));
        assert_eq!(evaluate(&eq("status", json!("inactive")), &data), Ok(false));
        assert_eq!(evaluate(&eq("user.role", json!("admin")), &data), Ok(true));
        let contains = ConditionExpr::Contains {
            field: "message".into(),
            value: "world".into(),
        };
        assert_eq!(evaluate(&contains, &data), Ok(true));
        let gt = |v| ConditionExpr::GreaterThan {
            field: "score".into(),
            value: v,
        };
        assert_eq!(evaluate(&gt(80.0), &data), Ok(true));
        assert_eq!(evaluate(&gt(90.0), &data), Ok(false));
        let exists = |f: &str| ConditionExpr::Exists { field: f.into() };
        assert_eq!(evaluate(&exists("name"), &data), Ok(true));
        assert_eq!(evaluate(&exists("missing"), &data), Ok(false));
        let and = ConditionExpr::And {
            conditions: vec![eq("a", json!(1)), eq("b", json!(2))],
        };
        assert_eq!(evaluate(&and, &data), Ok(true));
        let not = ConditionExpr::Not {
            condition: Box::new(eq("a", json!(99))),
        };
        assert_eq!(evaluate(&not, &data), Ok(true));
        let m = ConditionExpr::Matches {
            field: "status".into(),
            pattern: "^act".into(),
        };
        assert_eq!(evaluate(&m, &data), Ok(true));
        let bad = ConditionExpr::Matches {
            field: "status".into(),
            pattern: "(".into(),
        };
        assert!(matches!(
            evaluate(&bad, &data),
            Err(ConditionError::Regex { .. })
        ));
    }

    #[test]
    fn structured_serde_shape() {
        let e: ConditionExpr =
            serde_json::from_value(json!({"op": "greater_than", "field": "s", "value": 1.5}))
                .unwrap();
        assert_eq!(
            e,
            ConditionExpr::GreaterThan {
                field: "s".into(),
                value: 1.5
            }
        );
        let e: ConditionExpr = serde_json::from_value(
            json!({"op": "not", "condition": {"op": "in", "field": "k", "values": [1, "a"]}}),
        )
        .unwrap();
        assert!(matches!(e, ConditionExpr::Not { .. }));
    }

    #[test]
    fn null_and_missing_are_errors_except_exists() {
        let data = json!({"n": null});
        assert!(matches!(ev("n == 1", &data), Err(ConditionError::Field(_))));
        assert_eq!(ev("n", &data), Ok(false));
        assert_eq!(ev("missing", &data), Ok(false));
    }

    #[test]
    fn precedence_and_over_or() {
        // a && b || !c  ==  (a && b) || (!c)
        let e = parse("a && b || !c").unwrap();
        let ConditionExpr::Or { conditions } = e else {
            panic!("not or")
        };
        assert!(matches!(conditions[0], ConditionExpr::And { .. }));
        assert!(matches!(conditions[1], ConditionExpr::Not { .. }));
        let d = json!({"a": 1, "b": 1, "c": 1});
        assert_eq!(ev("a && b || !c", &d), Ok(true));
        assert_eq!(ev("a && (b || !c)", &d), Ok(true));
        assert_eq!(ev("!(a && b)", &d), Ok(false));
    }

    #[test]
    fn operators_and_literals() {
        let d = json!({"x": {"n": 3, "s": "it's", "f": 2.5, "t": true, "z": "a-b"}, "ids": [1]});
        assert_eq!(ev("x.n == 3", &d), Ok(true));
        assert_eq!(ev("x.n != 3", &d), Ok(false));
        assert_eq!(
            ev("x.n < 4 && x.n <= 3 && x.n > 2 && x.n >= 3", &d),
            Ok(true)
        );
        assert_eq!(ev("x.f > 2.4", &d), Ok(true));
        assert_eq!(ev("x.n > -1", &d), Ok(true));
        assert_eq!(ev("x.t == true", &d), Ok(true));
        assert_eq!(ev("x.s == \"it's\"", &d), Ok(true));
        assert_eq!(ev("x.s == 'it\\'s'", &d), Ok(true));
        assert_eq!(ev("x.z == 'a-b'", &d), Ok(true));
        assert_eq!(ev("x.n in [1, 2, 3]", &d), Ok(true));
        assert_eq!(ev("x.s in ['a', \"b\"]", &d), Ok(false));
        assert_eq!(ev("ids.0 == 1", &d), Ok(true));
    }

    #[test]
    fn string_escapes() {
        let e = parse(r"a == 'x\n\\\'y'").unwrap();
        assert_eq!(
            e,
            ConditionExpr::Equals {
                field: "a".into(),
                value: json!("x\n\\'y")
            }
        );
        assert!(parse(r"a == 'x\q'").is_err());
        assert!(parse("a == 'x").is_err());
    }

    #[test]
    fn parse_errors() {
        for bad in [
            "",
            "a ==",
            "a == b",
            "1 == a",
            "'x' == a",
            "a &&",
            "(a",
            "a)",
            "a in 1",
            "a in []",
            "a in [1",
            "a < 'x'",
            "a = 1",
            "a & b",
            "a @ b",
            "a b",
            "!",
            "a..b",
            "a == 1e999",
        ] {
            assert!(parse(bad).is_err(), "should fail: {bad:?}");
        }
        let deep = format!("{}a{}", "(".repeat(100), ")".repeat(100));
        assert!(parse(&deep).is_err());
        let nots = format!("{}a", "!".repeat(100));
        assert!(parse(&nots).is_err());
        assert!(parse(&"a ".repeat(3000)).is_err());
    }

    #[test]
    fn missing_field_propagates_through_not_ne_in() {
        let ctx = json!({"decide": {"action": "x"}});
        for src in [
            "!decide.missing == 'x'",
            "decide.missing != 'x'",
            "decide.missing in ['x']",
        ] {
            let out = when_holds(&Condition::Expr(src.into()), &ctx).unwrap();
            assert!(matches!(out, WhenOutcome::FieldError(_)), "{src}: {out:?}");
        }
        // `!` of a bare exists on a missing field is fine: exists never errors.
        let out = when_holds(&Condition::Expr("!decide.missing".into()), &ctx).unwrap();
        assert_eq!(out, WhenOutcome::True);
        let out = when_holds(&Condition::Expr("!(decide.missing == 'x')".into()), &ctx).unwrap();
        assert!(matches!(out, WhenOutcome::FieldError(_)));
        // operands run left to right: an erroring first operand errors the whole `||`
        let out = when_holds(
            &Condition::Expr("decide.missing == 1 || decide.action == 'x'".into()),
            &ctx,
        )
        .unwrap();
        assert!(matches!(out, WhenOutcome::FieldError(_)));
    }

    #[test]
    fn when_holds_true_false_and_errors() {
        let ctx = json!({"a": {"n": 2}});
        let t = |s: &str| when_holds(&Condition::Expr(s.into()), &ctx);
        assert_eq!(t("a.n == 2"), Ok(WhenOutcome::True));
        assert_eq!(t("a.n == 3"), Ok(WhenOutcome::False));
        assert!(matches!(t("a.n =="), Err(ConditionError::Parse(_))));
        let s = Condition::Structured(ConditionExpr::Exists { field: "a".into() });
        assert_eq!(when_holds(&s, &ctx), Ok(WhenOutcome::True));
    }

    #[test]
    fn short_circuit_order_decides() {
        let ctx = json!({"ok": "x"});
        // the left operand already decides, so the missing field is never read
        let t = |s: &str| when_holds(&Condition::Expr(s.into()), &ctx).unwrap();
        assert_eq!(t("ok == 'x' || missing == 1"), WhenOutcome::True);
        assert_eq!(t("ok == 'y' && missing == 1"), WhenOutcome::False);
        assert!(matches!(
            t("missing == 1 || ok == 'x'"),
            WhenOutcome::FieldError(_)
        ));
    }

    #[test]
    fn bare_path_is_exists_so_not_missing_is_true() {
        let ctx = json!({"decide": {"action": "x"}});
        assert_eq!(ev("!decide.missing", &ctx), Ok(true));
        assert_eq!(ev("decide.action", &ctx), Ok(true));
    }

    fn nested_not(depth: usize) -> ConditionExpr {
        let mut e = ConditionExpr::Exists { field: "a".into() };
        for _ in 0..depth {
            e = ConditionExpr::Not {
                condition: Box::new(e),
            };
        }
        e
    }

    #[test]
    fn bounds_depth_nodes_in_values_and_patterns() {
        assert!(check_bounds(&nested_not(MAX_COND_DEPTH - 1)).is_ok());
        assert!(check_bounds(&nested_not(MAX_COND_DEPTH + 1)).is_err());
        let leaf = ConditionExpr::Exists { field: "a".into() };
        let wide = |n| ConditionExpr::And {
            conditions: vec![leaf.clone(); n],
        };
        assert!(check_bounds(&wide(MAX_COND_NODES - 1)).is_ok());
        assert!(check_bounds(&wide(MAX_COND_NODES)).is_err());
        let ins = |n| ConditionExpr::In {
            field: "a".into(),
            values: vec![json!(1); n],
        };
        assert!(check_bounds(&ins(MAX_IN_VALUES)).is_ok());
        assert!(check_bounds(&ins(MAX_IN_VALUES + 1)).is_err());
        let pat = |p: String| ConditionExpr::Matches {
            field: "a".into(),
            pattern: p,
        };
        assert!(check_bounds(&pat("a".repeat(MAX_PATTERN_LEN))).is_ok());
        assert!(check_bounds(&pat("a".repeat(MAX_PATTERN_LEN + 1))).is_err());
        assert!(matches!(
            check_bounds(&pat("(".into())),
            Err(ConditionError::Regex { .. })
        ));
        // a pattern that compiles but exceeds the compiled-size limit
        assert!(check_bounds(&pat("((a{100}){100}){100}".into())).is_err());
    }

    #[test]
    fn parser_output_is_bounded_too() {
        let many = (0..=MAX_IN_VALUES)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert!(parse(&format!("a in [{many}]")).is_err());
        let wide = vec!["a"; MAX_COND_NODES + 1].join(" && ");
        assert!(parse(&wide).is_err());
    }

    #[test]
    fn evaluate_rejects_oversized_regex_at_run_time() {
        let e = ConditionExpr::Matches {
            field: "s".into(),
            pattern: "((a{100}){100}){100}".into(),
        };
        assert!(matches!(
            evaluate(&e, &json!({"s": "a"})),
            Err(ConditionError::Regex { .. })
        ));
    }
}
