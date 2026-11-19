//! Conversion of `tools/call` results into plain JSON values.

use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::Value;

use crate::error::{Error, Result};

/// Turns a `tools/call` result into a JSON value.
///
/// Rules, in order:
/// 1. `isError: true` → `Err`, carrying the trimmed, non-empty text blocks
///    joined by newlines (or a generic message when there is no text).
/// 2. Non-null `structuredContent` is authoritative and returned as-is.
/// 3. Exactly one text block → its text parsed as JSON when it parses
///    (scalars included), otherwise the text as a string (`""` stays `""`).
/// 4. Anything else (no blocks, several blocks, one non-text block) → the
///    content array as MCP JSON.
pub(crate) fn extract_tool_result(name: &str, r: CallToolResult) -> Result<Value> {
    if r.is_error == Some(true) {
        let text = r
            .content
            .iter()
            .filter_map(ContentBlock::as_text)
            .map(|t| t.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(Error::Mcp(if text.is_empty() {
            format!("MCP tool '{name}' reported an error without text content")
        } else {
            format!("MCP tool '{name}' error: {text}")
        }));
    }

    if let Some(structured) = r.structured_content
        && !structured.is_null()
    {
        return Ok(structured);
    }

    if let [block] = r.content.as_slice()
        && let Some(text) = block.as_text()
    {
        return Ok(serde_json::from_str::<Value>(&text.text)
            .unwrap_or_else(|_| Value::String(text.text.clone())));
    }

    Ok(serde_json::to_value(&r.content)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text(t: &str) -> ContentBlock {
        ContentBlock::text(t)
    }

    fn ok(r: CallToolResult) -> Value {
        extract_tool_result("t", r).unwrap()
    }

    fn err(r: CallToolResult) -> String {
        extract_tool_result("t", r).unwrap_err().to_string()
    }

    #[test]
    fn plain_text_becomes_string() {
        assert_eq!(
            ok(CallToolResult::success(vec![text("Hello")])),
            json!("Hello")
        );
    }

    #[test]
    fn json_text_is_parsed() {
        assert_eq!(
            ok(CallToolResult::success(vec![text("{\"a\":1}")])),
            json!({"a": 1})
        );
        assert_eq!(ok(CallToolResult::success(vec![text("3")])), json!(3));
    }

    #[test]
    fn is_error_with_text_names_the_tool() {
        let msg = err(CallToolResult::error(vec![
            text("  bad "),
            ContentBlock::image("AA==", "image/png"),
            text("second"),
        ]));
        assert!(msg.contains("bad\nsecond"), "{msg}");
        assert!(msg.contains("'t'"), "{msg}");
    }

    #[test]
    fn is_error_without_text() {
        let msg = err(CallToolResult::error(vec![]));
        assert!(
            msg.contains("reported an error without text content"),
            "{msg}"
        );
        assert!(msg.contains("'t'"), "{msg}");
    }

    #[test]
    fn structured_only() {
        let mut r = CallToolResult::success(vec![]);
        r.structured_content = Some(json!({"n": 1}));
        assert_eq!(ok(r), json!({"n": 1}));
        let mut scalar = CallToolResult::success(vec![]);
        scalar.structured_content = Some(json!(3));
        assert_eq!(ok(scalar), json!(3));
    }

    #[test]
    fn structured_wins_over_text() {
        let mut r = CallToolResult::success(vec![text("ignored")]);
        r.structured_content = Some(json!({"k": "v"}));
        assert_eq!(ok(r), json!({"k": "v"}));
    }

    #[test]
    fn null_structured_falls_back_to_text() {
        let mut r = CallToolResult::success(vec![text("plain")]);
        r.structured_content = Some(Value::Null);
        assert_eq!(ok(r), json!("plain"));
    }

    #[test]
    fn empty_text_stays_empty_string() {
        assert_eq!(ok(CallToolResult::success(vec![text("")])), json!(""));
    }

    #[test]
    fn several_blocks_return_content_array() {
        let v = ok(CallToolResult::success(vec![text("a"), text("b")]));
        assert_eq!(
            v,
            json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        );
    }

    #[test]
    fn single_non_text_block_returns_content_array() {
        let image = ContentBlock::image("AA==", "image/png");
        let expected = json!([serde_json::to_value(&image).unwrap()]);
        assert_eq!(ok(CallToolResult::success(vec![image])), expected);
        assert_eq!(expected[0]["type"], "image");
    }

    #[test]
    fn no_content_and_no_structured_is_empty_array() {
        assert_eq!(ok(CallToolResult::success(vec![])), json!([]));
    }

    #[test]
    fn is_error_false_is_success() {
        let mut r = CallToolResult::success(vec![text("fine")]);
        r.is_error = Some(false);
        assert_eq!(ok(r), json!("fine"));
    }
}
