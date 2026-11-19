use serde::{Deserialize, Serialize};

/// MCP transport protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpTransportType {
    /// Server-Sent Events (legacy).
    Sse,
    /// Streamable HTTP (recommended).
    HttpStream,
    /// Standard I/O (command-line process).
    Stdio,
}

/// Definition of an MCP tool, fetched from an MCP server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    pub output_schema: Option<serde_json::Value>,
}

/// Definition of an MCP resource.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpResourceDefinition {
    pub name: String,
    pub description: Option<String>,
    pub uri: String,
    pub input_schema: serde_json::Value,
    pub mime_type: Option<String>,
}

/// Definition of an MCP prompt template.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpPromptDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Service for fetching tool/resource/prompt definitions from MCP endpoints.
///
/// Connects to an MCP server via the configured transport and retrieves
/// available definitions. Mirrors Python's `MCPDefinitionService`.
pub struct McpDefinitionService {
    endpoint: String,
    transport_type: McpTransportType,
    working_directory: Option<String>,
}

impl McpDefinitionService {
    /// Creates a new MCP definition service.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - URL (for HTTP/SSE) or command string (for STDIO)
    /// * `transport_type` - Transport protocol to use
    pub fn new(endpoint: impl Into<String>, transport_type: McpTransportType) -> Self {
        Self {
            endpoint: endpoint.into(),
            transport_type,
            working_directory: None,
        }
    }

    /// Sets the working directory for STDIO commands.
    pub fn with_working_directory(mut self, dir: impl Into<String>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    /// Fetches tool definitions from the MCP server.
    ///
    /// Connects using the configured transport, calls `tools/list`
    /// (following `nextCursor` pagination), and returns the tool definitions
    /// with their input/output schemas.
    pub async fn fetch_tool_definitions(
        &self,
    ) -> Result<Vec<McpToolDefinition>, crate::error::Error> {
        let body = self.list_all("tools/list", "tools").await?;
        parse_tool_definitions(&body)
    }

    /// Fetches resource definitions from the MCP server.
    pub async fn fetch_resource_definitions(
        &self,
    ) -> Result<Vec<McpResourceDefinition>, crate::error::Error> {
        let body = self.list_all("resources/list", "resources").await?;
        parse_resource_definitions(&body)
    }

    /// Fetches prompt definitions from the MCP server.
    pub async fn fetch_prompt_definitions(
        &self,
    ) -> Result<Vec<McpPromptDefinition>, crate::error::Error> {
        let body = self.list_all("prompts/list", "prompts").await?;
        parse_prompt_definitions(&body)
    }

    /// Lists every page of `method`, returning a merged `{"result": {key: [..]}}` body.
    async fn list_all(
        &self,
        method: &'static str,
        key: &'static str,
    ) -> Result<serde_json::Value, crate::error::Error> {
        match self.transport_type {
            McpTransportType::HttpStream | McpTransportType::Sse => {
                collect_pages(key, |cursor| self.list_page_http(method, cursor)).await
            }
            McpTransportType::Stdio => self.list_all_stdio(method, key).await,
        }
    }

    // --- HTTP/SSE transport ---

    async fn list_page_http(
        &self,
        method: &str,
        cursor: Option<String>,
    ) -> Result<serde_json::Value, crate::error::Error> {
        let client = reqwest::Client::new();
        let url = transport_endpoint(&self.endpoint, &format!("/mcp/{method}"))?;

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&list_request(method, cursor.as_deref()))
            .send()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("HTTP request failed: {e}")))?;

        response
            .json()
            .await
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to parse response: {e}")))
    }

    // --- STDIO transport ---

    /// Pages through `method` using a single child process, so continuation
    /// cursors reach the process that issued them.
    async fn list_all_stdio(
        &self,
        method: &'static str,
        key: &'static str,
    ) -> Result<serde_json::Value, crate::error::Error> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        use tokio::process::Command;

        let parts: Vec<&str> = self.endpoint.split_whitespace().collect();
        if parts.is_empty() {
            return Err(crate::error::Error::Mcp(
                "STDIO endpoint command is empty".to_string(),
            ));
        }

        let mut cmd = Command::new(parts[0]);
        if parts.len() > 1 {
            cmd.args(&parts[1..]);
        }

        if let Some(dir) = &self.working_directory {
            cmd.current_dir(dir);
        }

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());
        cmd.kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| crate::error::Error::Mcp(format!("Failed to spawn process: {e}")))?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(crate::error::Error::Mcp(
                "Failed to open STDIO pipes".to_string(),
            ));
        };

        // (stdin, stdout line reader, next request id)
        let io = tokio::sync::Mutex::new((stdin, tokio::io::BufReader::new(stdout).lines(), 1u64));
        let result = collect_pages(key, |cursor| {
            let io = &io;
            async move {
                let mut guard = io.lock().await;
                let (stdin, lines, next_id) = &mut *guard;
                let mut request = list_request(method, cursor.as_deref());
                request["id"] = serde_json::json!(*next_id);
                *next_id += 1;
                let mut line = request.to_string();
                line.push('\n');
                stdin.write_all(line.as_bytes()).await.map_err(|e| {
                    crate::error::Error::Mcp(format!("Failed to write to stdin: {e}"))
                })?;
                stdin.flush().await.map_err(|e| {
                    crate::error::Error::Mcp(format!("Failed to write to stdin: {e}"))
                })?;
                let response = lines
                    .next_line()
                    .await
                    .map_err(|e| crate::error::Error::Mcp(format!("Process failed: {e}")))?
                    .ok_or_else(|| {
                        crate::error::Error::Mcp(
                            "STDIO process closed stdout before responding".to_string(),
                        )
                    })?;
                serde_json::from_str(&response).map_err(|e| {
                    crate::error::Error::Mcp(format!("Failed to parse STDIO output: {e}"))
                })
            }
        })
        .await;

        // Close stdin (EOF) and reap the child.
        let (stdin, _, _) = io.into_inner();
        drop(stdin);
        let status = child.wait().await;
        let value = result?;
        status.map_err(|e| crate::error::Error::Mcp(format!("Process failed: {e}")))?;
        Ok(value)
    }
}

/// Errors raised while paginating an MCP list response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpPaginationError {
    /// The server returned a `nextCursor` it had already returned.
    RepeatedCursor(String),
    /// A page did not contain the expected result array.
    MissingItems(&'static str),
}

impl std::fmt::Display for McpPaginationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepeatedCursor(c) => {
                write!(f, "MCP pagination returned a repeated cursor: {c:?}")
            }
            Self::MissingItems(key) => write!(f, "No {key} in response"),
        }
    }
}

impl std::error::Error for McpPaginationError {}

impl From<McpPaginationError> for crate::error::Error {
    fn from(e: McpPaginationError) -> Self {
        crate::error::Error::Mcp(e.to_string())
    }
}

/// Builds a JSON-RPC list request, adding `params.cursor` for continuation pages.
fn list_request(method: &str, cursor: Option<&str>) -> serde_json::Value {
    let mut request = serde_json::json!({"jsonrpc": "2.0", "method": method, "id": 1});
    if let Some(cursor) = cursor {
        request["params"] = serde_json::json!({"cursor": cursor});
    }
    request
}

/// Follows opaque `nextCursor` values, merging every page's `result[key]` array.
///
/// Stops with [`McpPaginationError::RepeatedCursor`] if a cursor repeats, so a
/// misbehaving server cannot cause an infinite loop. Errors from `fetch`
/// (including continuation-page failures) propagate unchanged.
async fn collect_pages<F, Fut>(
    key: &'static str,
    mut fetch: F,
) -> Result<serde_json::Value, crate::error::Error>
where
    F: FnMut(Option<String>) -> Fut,
    Fut: std::future::Future<Output = Result<serde_json::Value, crate::error::Error>>,
{
    let mut items: Vec<serde_json::Value> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut cursor: Option<String> = None;
    loop {
        let page = fetch(cursor.take()).await?;
        let result = page.get("result");
        let page_items = result
            .and_then(|r| r.get(key))
            .and_then(|t| t.as_array())
            .ok_or(McpPaginationError::MissingItems(key))?;
        items.extend(page_items.iter().cloned());
        match result
            .and_then(|r| r.get("nextCursor"))
            .and_then(|c| c.as_str())
        {
            Some(next) => {
                if !seen.insert(next.to_string()) {
                    return Err(McpPaginationError::RepeatedCursor(next.to_string()).into());
                }
                cursor = Some(next.to_string());
            }
            None => break,
        }
    }
    Ok(serde_json::json!({"result": {key: items}}))
}

/// Method paths the module addresses, used to recognise explicit method URLs.
const KNOWN_METHOD_SUFFIXES: [&str; 4] = [
    "/mcp/tools/list",
    "/mcp/tools/call",
    "/mcp/resources/list",
    "/mcp/prompts/list",
];

/// Resolves an HTTP endpoint for `suffix` (e.g. `/mcp/tools/list`).
///
/// An endpoint that already ends with `suffix` is used as-is. If it ends with a
/// different known `/mcp/<method>` suffix, that suffix is replaced; otherwise
/// `suffix` is appended to the path. Query parameters and fragments are preserved.
pub(crate) fn transport_endpoint(
    endpoint: &str,
    suffix: &str,
) -> Result<String, crate::error::Error> {
    let mut url = reqwest::Url::parse(endpoint)
        .map_err(|e| crate::error::Error::Mcp(format!("Invalid endpoint URL: {e}")))?;
    let path = url.path().trim_end_matches('/').to_string();
    if path.ends_with(suffix.trim_end_matches('/')) {
        return Ok(endpoint.to_string());
    }
    let base = KNOWN_METHOD_SUFFIXES
        .iter()
        .find_map(|known| path.strip_suffix(known))
        .unwrap_or(&path);
    url.set_path(&format!("{base}{suffix}"));
    Ok(url.to_string())
}

fn parse_tool_definitions(
    body: &serde_json::Value,
) -> Result<Vec<McpToolDefinition>, crate::error::Error> {
    let tools = body
        .get("result")
        .and_then(|r| r.get("tools"))
        .and_then(|t| t.as_array())
        .ok_or_else(|| crate::error::Error::Mcp("No tools in response".to_string()))?;

    let mut definitions = Vec::new();
    for tool in tools {
        definitions.push(McpToolDefinition {
            name: tool
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            description: tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(String::from),
            input_schema: tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(serde_json::json!({})),
            output_schema: tool.get("outputSchema").cloned(),
        });
    }
    Ok(definitions)
}

fn parse_resource_definitions(
    body: &serde_json::Value,
) -> Result<Vec<McpResourceDefinition>, crate::error::Error> {
    let resources = body
        .get("result")
        .and_then(|r| r.get("resources"))
        .and_then(|t| t.as_array())
        .ok_or_else(|| crate::error::Error::Mcp("No resources in response".to_string()))?;

    let mut definitions = Vec::new();
    for res in resources {
        definitions.push(McpResourceDefinition {
            name: res
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            description: res
                .get("description")
                .and_then(|d| d.as_str())
                .map(String::from),
            uri: res
                .get("uri")
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string(),
            input_schema: res
                .get("inputSchema")
                .cloned()
                .unwrap_or(serde_json::json!({})),
            mime_type: res
                .get("mimeType")
                .and_then(|m| m.as_str())
                .map(String::from),
        });
    }
    Ok(definitions)
}

fn parse_prompt_definitions(
    body: &serde_json::Value,
) -> Result<Vec<McpPromptDefinition>, crate::error::Error> {
    let prompts = body
        .get("result")
        .and_then(|r| r.get("prompts"))
        .and_then(|t| t.as_array())
        .ok_or_else(|| crate::error::Error::Mcp("No prompts in response".to_string()))?;

    let mut definitions = Vec::new();
    for prompt in prompts {
        definitions.push(McpPromptDefinition {
            name: prompt
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            description: prompt
                .get("description")
                .and_then(|d| d.as_str())
                .map(String::from),
            input_schema: prompt
                .get("inputSchema")
                .cloned()
                .unwrap_or(serde_json::json!({})),
        });
    }
    Ok(definitions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_definitions() {
        let body = serde_json::json!({
            "result": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        });

        let defs = parse_tool_definitions(&body).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "search");
        assert_eq!(defs[0].description.as_deref(), Some("Search the web"));
    }

    #[test]
    fn test_parse_resource_definitions() {
        let body = serde_json::json!({
            "result": {
                "resources": [
                    {
                        "name": "config",
                        "description": "App configuration",
                        "uri": "config://app",
                        "mimeType": "application/json",
                        "inputSchema": {}
                    }
                ]
            }
        });

        let defs = parse_resource_definitions(&body).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "config");
        assert_eq!(defs[0].uri, "config://app");
    }

    #[test]
    fn test_parse_prompt_definitions() {
        let body = serde_json::json!({
            "result": {
                "prompts": [
                    {
                        "name": "summarize",
                        "description": "Summarize text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        });

        let defs = parse_prompt_definitions(&body).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "summarize");
    }

    use std::sync::Mutex;

    fn page(key: &str, items: &[&str], next: Option<&str>) -> serde_json::Value {
        let list: Vec<_> = items
            .iter()
            .map(|n| serde_json::json!({"name": n, "uri": "x"}))
            .collect();
        let mut result = serde_json::json!({ key: list });
        if let Some(n) = next {
            result["nextCursor"] = serde_json::json!(n);
        }
        serde_json::json!({"result": result})
    }

    #[tokio::test]
    async fn test_collect_pages_follows_cursors() {
        let seen = Mutex::new(Vec::new());
        let body = collect_pages("tools", |cursor| {
            seen.lock().unwrap().push(cursor.clone());
            let p = match cursor.as_deref() {
                None => page("tools", &["a"], Some("c1")),
                Some("c1") => page("tools", &["b"], Some("c2")),
                _ => page("tools", &["c"], None),
            };
            async move { Ok(p) }
        })
        .await
        .unwrap();
        let defs = parse_tool_definitions(&body).unwrap();
        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert_eq!(
            *seen.lock().unwrap(),
            vec![None, Some("c1".to_string()), Some("c2".to_string())]
        );
    }

    #[tokio::test]
    async fn test_collect_pages_stops_on_repeated_cursor() {
        let err = collect_pages("tools", |_| async {
            Ok(page("tools", &["a"], Some("same")))
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("repeated cursor"), "{err}");
    }

    #[tokio::test]
    async fn test_collect_pages_propagates_continuation_error() {
        let err = collect_pages("prompts", |cursor| async move {
            match cursor {
                None => Ok(page("prompts", &["a"], Some("c1"))),
                Some(_) => Err(crate::error::Error::Mcp("boom".into())),
            }
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("boom"));
    }

    #[tokio::test]
    async fn test_collect_pages_missing_items() {
        let err = collect_pages("tools", |_| async { Ok(serde_json::json!({"result": {}})) })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("No tools in response"));
    }

    #[test]
    fn test_list_request_cursor() {
        assert!(list_request("tools/list", None).get("params").is_none());
        assert_eq!(
            list_request("tools/list", Some("abc"))["params"]["cursor"],
            "abc"
        );
    }

    #[test]
    fn test_transport_endpoint_replaces_other_method_suffix() {
        let call = "/mcp/tools/call";
        let list = "/mcp/tools/list";
        assert_eq!(
            transport_endpoint("https://host/api/mcp/tools/list", call).unwrap(),
            "https://host/api/mcp/tools/call"
        );
        assert_eq!(
            transport_endpoint("https://host/api/mcp/tools/call", list).unwrap(),
            "https://host/api/mcp/tools/list"
        );
        assert_eq!(
            transport_endpoint("https://host/api/mcp/tools/list?t=a%2Fb#f", call).unwrap(),
            "https://host/api/mcp/tools/call?t=a%2Fb#f"
        );
        assert_eq!(
            transport_endpoint("https://host/api/mcp/resources/list", "/mcp/prompts/list").unwrap(),
            "https://host/api/mcp/prompts/list"
        );
        assert_eq!(
            transport_endpoint("https://host/api", call).unwrap(),
            "https://host/api/mcp/tools/call"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_stdio_pagination_uses_single_process() {
        let script = r#"issued=0
while IFS= read -r line; do
  case "$line" in
    *'"cursor":"c1"'*)
      if [ "$issued" = 1 ]; then
        echo '{"result":{"tools":[{"name":"b"}]}}'
      else
        echo '{"error":{"message":"unknown cursor"}}'
      fi;;
    *)
      issued=1
      echo '{"result":{"tools":[{"name":"a"}],"nextCursor":"c1"}}';;
  esac
done
"#;
        let path =
            std::env::temp_dir().join(format!("mcp_stdio_fixture_{}.sh", std::process::id()));
        std::fs::write(&path, script).unwrap();
        let svc =
            McpDefinitionService::new(format!("sh {}", path.display()), McpTransportType::Stdio);
        let result = svc.fetch_tool_definitions().await;
        let _ = std::fs::remove_file(&path);
        let names: Vec<String> = result.unwrap().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_transport_endpoint() {
        let s = "/mcp/tools/list";
        assert_eq!(
            transport_endpoint("https://example.com", s).unwrap(),
            "https://example.com/mcp/tools/list"
        );
        assert_eq!(
            transport_endpoint("https://example.com/api/", s).unwrap(),
            "https://example.com/api/mcp/tools/list"
        );
        assert_eq!(
            transport_endpoint("https://example.com/api/mcp/tools/list", s).unwrap(),
            "https://example.com/api/mcp/tools/list"
        );
        assert_eq!(
            transport_endpoint("https://example.com/api/mcp/tools/list/?token=a%2Fb#s", s).unwrap(),
            "https://example.com/api/mcp/tools/list/?token=a%2Fb#s"
        );
        assert_eq!(
            transport_endpoint("https://example.com?token=a%2Fb#section", s).unwrap(),
            "https://example.com/mcp/tools/list?token=a%2Fb#section"
        );
        assert!(transport_endpoint("not a url", s).is_err());
    }

    #[test]
    fn test_resource_uri_identity_preserved() {
        let uri = "file:///a%20b/%7Bnot-a-template%7D?x=1%2F2";
        let body = serde_json::json!({"result": {"resources": [{"name": "r", "uri": uri}]}});
        assert_eq!(parse_resource_definitions(&body).unwrap()[0].uri, uri);
    }

    #[test]
    fn test_input_schemas_pass_through_unchanged() {
        let schemas = [
            serde_json::json!(true),
            serde_json::json!({"$ref": "#/$defs/a~1b", "$defs": {"a/b": {"type": "string"}}}),
            serde_json::json!({"type": "string", "enum": ["a", 1], "const": "a"}),
            serde_json::json!({"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "minItems": 2, "maxItems": 2}),
            serde_json::json!({"type": "integer", "minimum": 1, "exclusiveMaximum": 9, "multipleOf": 2}),
            serde_json::json!({"type": "object", "additionalProperties": {"type": "integer"}, "minProperties": 1, "properties": {"tool_name": {"type": "string"}}}),
            serde_json::json!({"$defs": {"n": {"type": "object", "properties": {"c": {"$ref": "#/$defs/n"}}}}, "$ref": "#/$defs/n"}),
        ];
        for schema in schemas {
            let body =
                serde_json::json!({"result": {"tools": [{"name": "t", "inputSchema": schema}]}});
            assert_eq!(
                parse_tool_definitions(&body).unwrap()[0].input_schema,
                schema
            );
        }
    }
}
