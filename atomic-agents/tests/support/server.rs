//! In-process MCP server used by the integration harness.
//!
//! The same handler backs every transport: it is served over stdio when the
//! harness binary is re-executed as a child process, and over Streamable HTTP
//! on a loopback listener. Per-method counters let scenarios assert exactly
//! how many requests reached the server (for example, one `initialize` per
//! session, and zero `prompts/list` when the capability is not advertised).

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    model::*,
    service::{MaybeSendFuture, RequestContext},
};
use serde_json::{Value, json};

/// Tools-per-page for `tools/list`, small enough to force several pages.
pub const TOOLS_PAGE_SIZE: usize = 2;

/// Cursor the repeating-cursor variant hands out for every continuation page.
const LOOP_CURSOR: &str = "loop";

/// Requests received by a [`TestServer`], shared by every clone of it.
#[derive(Debug, Default)]
pub struct Counters {
    pub initialize: AtomicUsize,
    pub tools_list: AtomicUsize,
    pub resources_list: AtomicUsize,
    pub prompts_list: AtomicUsize,
}

impl Counters {
    /// The counters as JSON, in the shape the `stats` tool returns.
    pub fn snapshot(&self) -> Value {
        json!({
            "initialize": self.initialize.load(Ordering::SeqCst),
            "tools_list": self.tools_list.load(Ordering::SeqCst),
            "resources_list": self.resources_list.load(Ordering::SeqCst),
            "prompts_list": self.prompts_list.load(Ordering::SeqCst),
        })
    }
}

/// A configurable MCP test server.
#[derive(Clone, Default)]
pub struct TestServer {
    counters: Arc<Counters>,
    tools_only: bool,
    repeat_cursor: bool,
    no_templates: bool,
}

impl TestServer {
    /// A server advertising only the `tools` capability.
    pub fn tools_only() -> Self {
        Self {
            tools_only: true,
            ..Self::default()
        }
    }

    /// A server whose `tools/list` pagination hands out the same cursor twice.
    pub fn repeating_cursor() -> Self {
        Self {
            repeat_cursor: true,
            ..Self::default()
        }
    }

    /// A server that answers `resources/templates/list` with `METHOD_NOT_FOUND`
    /// while still advertising `resources`.
    pub fn without_templates() -> Self {
        Self {
            no_templates: true,
            ..Self::default()
        }
    }

    /// The request counters shared by this server and its clones.
    pub fn counters(&self) -> Arc<Counters> {
        self.counters.clone()
    }
}

fn object(v: Value) -> JsonObject {
    v.as_object().expect("schema literal is an object").clone()
}

/// Every tool the server exposes, in listing order.
pub fn all_tools() -> Vec<Tool> {
    let empty = || object(json!({"type": "object", "properties": {}}));
    vec![
        Tool::new(
            "echo",
            "Echo text back",
            object(json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            })),
        ),
        Tool::new(
            "add",
            "Add two numbers",
            object(json!({
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]
            })),
        )
        .with_raw_output_schema(Arc::new(object(json!({
            "type": "object",
            "properties": {"sum": {"type": "number"}}
        })))),
        Tool::new("boom", "Always reports an error", empty()),
        Tool::new("json_text", "Returns JSON encoded as text", empty()),
        Tool::new("stats", "Returns the server's request counters", empty()),
        Tool::new(
            "sleep",
            "Sleeps for `ms` milliseconds, then returns",
            object(json!({"type": "object", "properties": {"ms": {"type": "number"}}})),
        ),
        Tool::new(
            "env",
            "Reports whether environment variable `name` is set in the server process",
            object(json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            })),
        ),
    ]
}

fn number_arg(request: &CallToolRequestParams, key: &str) -> f64 {
    request
        .arguments
        .as_ref()
        .and_then(|a| a.get(key))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerConfig {
        let caps = if self.tools_only {
            ServerCapabilities::builder().enable_tools().build()
        } else {
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_prompts()
                .build()
        };
        ServerConfig::new(caps).with_server_info(Implementation::new("test-server", "0.0.1"))
    }

    fn initialize(
        &self,
        request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<InitializeResult, McpError>> + MaybeSendFuture + '_ {
        self.counters.initialize.fetch_add(1, Ordering::SeqCst);
        context.peer.set_peer_info(request.clone());
        std::future::ready(self.negotiate_initialize(&request))
    }

    // Paginated: the cursor is the start index (or `"loop"` on the
    // repeating-cursor variant, which keeps returning it).
    fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, McpError>> + MaybeSendFuture + '_ {
        self.counters.tools_list.fetch_add(1, Ordering::SeqCst);
        let cursor = request.and_then(|r| r.cursor);
        let all = all_tools();
        let start = match cursor.as_deref() {
            None => 0,
            Some(LOOP_CURSOR) => TOOLS_PAGE_SIZE,
            Some(c) => c.parse().unwrap_or(all.len()),
        }
        .min(all.len());
        let end = (start + TOOLS_PAGE_SIZE).min(all.len());
        let mut res = ListToolsResult::with_all_items(all[start..end].to_vec());
        res.next_cursor = if self.repeat_cursor {
            Some(LOOP_CURSOR.to_string())
        } else if end < all.len() {
            Some(end.to_string())
        } else {
            None
        };
        std::future::ready(Ok(res))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<CallToolResponse, McpError>> + MaybeSendFuture + '_ {
        let delay = (request.name == "sleep")
            .then(|| std::time::Duration::from_millis(number_arg(&request, "ms") as u64));
        let result = match request.name.as_ref() {
            "echo" => {
                let text = request
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
            }
            "add" => {
                let sum = number_arg(&request, "a") + number_arg(&request, "b");
                Ok(CallToolResult::structured(json!({"sum": sum})))
            }
            "boom" => Ok(CallToolResult::error(vec![ContentBlock::text("it broke")])),
            "json_text" => Ok(CallToolResult::success(vec![ContentBlock::text(
                "{\"k\":1}",
            )])),
            "stats" => Ok(CallToolResult::structured(self.counters.snapshot())),
            "sleep" => Ok(CallToolResult::success(vec![ContentBlock::text("slept")])),
            "env" => {
                let name = request
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let value = std::env::var(name).ok();
                Ok(CallToolResult::structured(
                    json!({"set": value.is_some(), "value": value}),
                ))
            }
            other => Err(McpError::invalid_params(
                format!("unknown tool {other}"),
                None,
            )),
        };
        async move {
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            result.map(CallToolResponse::from)
        }
    }

    fn list_resources(
        &self,
        _r: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourcesResult, McpError>> + MaybeSendFuture + '_ {
        self.counters.resources_list.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(ListResourcesResult::with_all_items(vec![
            Resource::new("mem://hello", "hello").with_mime_type("text/plain"),
            Resource::new("mem://blob", "blob").with_mime_type("application/octet-stream"),
        ])))
    }

    fn list_resource_templates(
        &self,
        _r: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourceTemplatesResult, McpError>> + MaybeSendFuture + '_
    {
        if self.no_templates {
            return std::future::ready(Err(McpError::method_not_found::<
                ListResourceTemplatesRequestMethod,
            >()));
        }
        std::future::ready(Ok(ListResourceTemplatesResult::with_all_items(vec![
            ResourceTemplate::new("mem://notes/{id}", "note").with_mime_type("text/plain"),
        ])))
    }

    fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ReadResourceResponse, McpError>> + MaybeSendFuture + '_ {
        let uri = request.uri.clone();
        let contents = match uri.as_str() {
            "mem://hello" => Some(ResourceContents::text("hi", &uri).with_mime_type("text/plain")),
            "mem://blob" => Some(
                ResourceContents::blob("AAEC", &uri).with_mime_type("application/octet-stream"),
            ),
            other => other
                .strip_prefix("mem://notes/")
                .map(|id| ResourceContents::text(format!("note {id}"), &uri)),
        };
        std::future::ready(match contents {
            Some(c) => Ok(ReadResourceResult::new(vec![c]).into()),
            None => Err(McpError::resource_not_found(
                format!("no resource {uri}"),
                None,
            )),
        })
    }

    fn list_prompts(
        &self,
        _r: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListPromptsResult, McpError>> + MaybeSendFuture + '_ {
        self.counters.prompts_list.fetch_add(1, Ordering::SeqCst);
        let mut name = PromptArgument::new("name").with_description("Who to greet");
        name.required = Some(true);
        std::future::ready(Ok(ListPromptsResult::with_all_items(vec![Prompt::new(
            "greet",
            Some("Greets someone"),
            Some(vec![name]),
        )])))
    }

    fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<GetPromptResponse, McpError>> + MaybeSendFuture + '_ {
        let result = if request.name == "greet" {
            match request
                .arguments
                .as_ref()
                .and_then(|a| a.get("name"))
                .and_then(Value::as_str)
            {
                Some(name) => Ok(GetPromptResult::new(vec![PromptMessage::new_text(
                    Role::User,
                    format!("Hello, {name}!"),
                )])
                .with_description("Greets someone")
                .into()),
                None => Err(McpError::invalid_params(
                    "missing required argument 'name'",
                    None,
                )),
            }
        } else {
            Err(McpError::invalid_params(
                format!("unknown prompt {}", request.name),
                None,
            ))
        };
        std::future::ready(result)
    }
}
