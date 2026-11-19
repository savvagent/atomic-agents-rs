//! End-to-end tests for the MCP client against an in-process rmcp server.
//!
//! This target uses `harness = false` (see `Cargo.toml`) because the stdio
//! scenarios re-execute this very binary as the MCP server child process
//! (`--mcp-test-server-stdio`). Under libtest the child would print test
//! banners to stdout, corrupting the JSON-RPC stream, and a dedicated
//! `[[bin]]` cannot use dev-dependencies. `main` therefore either acts as the
//! stdio server or runs the scenario list below.
//!
//! Usage: `cargo test -p atomic-agents --features mcp --test mcp_integration [filter]`.
//! A single positional argument filters scenarios by substring; arguments
//! starting with `-` are ignored. Every scenario is bounded by a 30 s timeout
//! and uses loopback only.

mod support;

use std::time::Duration;

use atomic_agents::agent::{AgentConfig, AtomicAgent};
use atomic_agents::error::Error;
use atomic_agents::mcp::{
    McpClient, McpDefinitionService, McpOrchestratorOutput, McpPrompt, McpPromptOutput,
    McpResource, McpResourceContent, McpResourceOutput, McpTool, McpToolCall, McpToolInput,
    McpToolOutput, McpToolset, McpTransport, McpTransportType, fetch_mcp_attributes_with_schema,
};
use atomic_agents::prompt::Prompt;
use atomic_agents::resource::Resource;
use atomic_agents::schema::BasicChatInputSchema;
use atomic_agents::tool::Tool;
use futures::FutureExt;
use futures::future::BoxFuture;
use rmcp::ServiceExt;
use serde_json::{Map, Value, json};
use support::legacy_sse_server::{LegacyOptions, spawn_legacy_sse};
use support::scripted_llm::ScriptedLlm;
use support::server::{TOOLS_PAGE_SIZE, all_tools};
use support::{TestServer, spawn_http};

const STDIO_FLAG: &str = "--mcp-test-server-stdio";
const STDIO_TOOLS_ONLY_FLAG: &str = "--mcp-test-server-stdio-tools-only";
const SCENARIO_TIMEOUT: Duration = Duration::from_secs(30);

type Scenario = (&'static str, fn() -> BoxFuture<'static, ()>);

fn scenarios() -> Vec<Scenario> {
    vec![
        ("stdio_discovery_and_handshake", || {
            stdio_discovery_and_handshake().boxed()
        }),
        ("http_discovery_and_handshake", || {
            http_discovery_and_handshake().boxed()
        }),
        ("http_endpoint_query_preserved", || {
            http_endpoint_query_preserved().boxed()
        }),
        ("capability_gating_tools_only", || {
            capability_gating_tools_only().boxed()
        }),
        ("repeated_cursor_errors", || {
            repeated_cursor_errors().boxed()
        }),
        ("http_connect_error_redacts_query", || {
            http_connect_error_redacts_query().boxed()
        }),
        ("http_transport_logs_redact_query", || {
            http_transport_logs_redact_query().boxed()
        }),
        ("closed_session_errors", || closed_session_errors().boxed()),
        ("tool_trait_over_http", || tool_trait_over_http().boxed()),
        ("tool_trait_over_stdio", || tool_trait_over_stdio().boxed()),
        ("tool_run_on_current_thread_runtime_errors", || {
            tool_run_on_current_thread_runtime_errors().boxed()
        }),
        ("resources_over_http", || resources_over_http().boxed()),
        ("resources_over_stdio", || resources_over_stdio().boxed()),
        ("resource_templates_method_not_found_tolerated", || {
            resource_templates_method_not_found_tolerated().boxed()
        }),
        ("resource_read_error_redacts_query", || {
            resource_read_error_redacts_query().boxed()
        }),
        ("resource_read_on_current_thread_runtime_errors", || {
            resource_read_on_current_thread_runtime_errors().boxed()
        }),
        ("prompts_over_http", || prompts_over_http().boxed()),
        ("prompts_over_stdio", || prompts_over_stdio().boxed()),
        ("prompt_generate_on_current_thread_runtime_errors", || {
            prompt_generate_on_current_thread_runtime_errors().boxed()
        }),
        ("prompt_get_error_redacts_query", || {
            prompt_get_error_redacts_query().boxed()
        }),
        ("definition_service_one_shot", || {
            definition_service_one_shot().boxed()
        }),
        ("orchestrator_selects_mcp_tool", || {
            orchestrator_selects_mcp_tool().boxed()
        }),
        ("orchestrator_unknown_tool_dispatch_errors", || {
            orchestrator_unknown_tool_dispatch_errors().boxed()
        }),
        ("sse_discovery_calls_resources_prompts", || {
            sse_discovery_calls_resources_prompts().boxed()
        }),
        ("sse_cross_origin_endpoint_rejected", || {
            sse_cross_origin_endpoint_rejected().boxed()
        }),
        ("sse_header_forwarded", || sse_header_forwarded().boxed()),
        ("sse_cancelled_connect_closes_stream", || {
            sse_cancelled_connect_closes_stream().boxed()
        }),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == STDIO_FLAG) {
        serve_stdio(TestServer::default());
    }
    if args.iter().any(|a| a == STDIO_TOOLS_ONLY_FLAG) {
        serve_stdio(TestServer::tools_only());
    }

    // Libtest-style flags: `--flag value` pairs are skipped as a unit.
    let mut filter = None;
    let mut skips: Vec<String> = Vec::new();
    let mut list = false;
    let mut ignored = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--list" => list = true,
            // `--include-ignored` lists/runs everything, like libtest.
            "--ignored" => ignored = true,
            "--skip" => skips.extend(it.next().cloned()),
            "--test-threads" | "--format" | "--color" => {
                it.next();
            }
            _ if a.starts_with('-') => {}
            _ => filter = filter.or_else(|| Some(a.clone())),
        }
    }
    if list {
        // No scenario is marked ignored.
        for (name, _) in scenarios().into_iter().filter(|_| !ignored) {
            println!("{name}: test");
        }
        std::process::exit(0);
    }
    if let Some(f) = &filter
        && !scenarios().iter().any(|(n, _)| n.contains(f.as_str()))
    {
        // Same as libtest: `cargo test <filter>` hands the filter to every test
        // binary, so a filter aimed at unit tests must not fail this one. Say so
        // loudly instead, so a typo'd filter is not mistaken for a green run.
        eprintln!("note: filter {f:?} matched no MCP integration scenarios; nothing ran");
        println!("0 passed, 0 failed (0 matched filter {f:?})");
        std::process::exit(0);
    }
    support::logs::install();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let (mut passed, mut failed) = (0usize, 0usize);
    for (name, scenario) in scenarios() {
        if filter.as_deref().is_some_and(|f| !name.contains(f))
            || skips.iter().any(|k| name.contains(k.as_str()))
        {
            continue;
        }
        // Spawning isolates panics: a panicking scenario surfaces as a JoinError.
        let outcome = runtime.block_on(async move {
            match tokio::spawn(tokio::time::timeout(SCENARIO_TIMEOUT, scenario())).await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(_)) => Err(format!("timed out after {SCENARIO_TIMEOUT:?}")),
                Err(e) => Err(format!("panicked: {e}")),
            }
        });
        match outcome {
            Ok(()) => {
                passed += 1;
                println!("test {name} ... ok");
            }
            Err(why) => {
                failed += 1;
                println!("test {name} ... FAILED ({why})");
            }
        }
    }
    println!("{passed} passed, {failed} failed");
    if failed > 0 {
        std::process::exit(1);
    }
}

/// Runs `server` over this process's stdin/stdout until the client hangs up.
fn serve_stdio(server: TestServer) -> ! {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let code = runtime.block_on(async move {
        match server.serve(rmcp::transport::stdio()).await {
            Ok(running) => i32::from(running.waiting().await.is_err()),
            Err(_) => 1,
        }
    });
    std::process::exit(code)
}

fn stdio_transport(flag: &str) -> McpTransport {
    let exe = std::env::current_exe().expect("current exe");
    McpTransport::stdio(exe.to_string_lossy(), vec![flag.to_string()])
}

fn args(v: Value) -> Map<String, Value> {
    v.as_object()
        .expect("arguments literal is an object")
        .clone()
}

fn assert_err_contains<T: std::fmt::Debug>(r: Result<T, Error>, needle: &str) {
    match r {
        Ok(v) => panic!("expected an error containing {needle:?}, got Ok({v:?})"),
        Err(e) => assert!(e.to_string().contains(needle), "{e} lacks {needle:?}"),
    }
}

fn tools_pages() -> usize {
    all_tools().len().div_ceil(TOOLS_PAGE_SIZE)
}

/// Discovery returns every page in server order with schemas untouched.
async fn assert_discovery(client: &McpClient) {
    let defs = client.list_tools().await.expect("list_tools");
    let expected = all_tools();
    let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    let expected_names: Vec<&str> = expected.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(names, expected_names);
    for (def, tool) in defs.iter().zip(&expected) {
        assert_eq!(
            def.input_schema,
            Value::Object((*tool.input_schema).clone())
        );
        assert_eq!(
            def.output_schema,
            tool.output_schema
                .as_ref()
                .map(|s| Value::Object((**s).clone()))
        );
        assert_eq!(def.description.as_deref(), tool.description.as_deref());
    }
}

/// Text, structured, isError and unknown-tool calls.
async fn assert_calls(client: &McpClient) {
    let echo = client.call_tool("echo", args(json!({"text": "hi"}))).await;
    assert_eq!(echo.expect("echo"), json!("hi"));
    let add = client
        .call_tool("add", args(json!({"a": 2, "b": 3})))
        .await
        .expect("add");
    assert_eq!(add["sum"].as_f64(), Some(5.0), "{add}");
    assert_err_contains(client.call_tool("boom", Map::new()).await, "it broke");
    assert!(client.call_tool("no_such_tool", Map::new()).await.is_err());
}

/// Discovery twice plus several calls over one session; returns `stats`.
async fn exercise_session(client: &McpClient) -> Value {
    assert_discovery(client).await;
    assert_calls(client).await;
    assert_discovery(client).await;
    let json_text = client.call_tool("json_text", Map::new()).await;
    assert_eq!(json_text.expect("json_text"), json!({"k": 1}));
    assert_calls(client).await;
    client.call_tool("stats", Map::new()).await.expect("stats")
}

async fn stdio_discovery_and_handshake() {
    let client = McpClient::connect(stdio_transport(STDIO_FLAG))
        .await
        .expect("connect stdio");
    assert_eq!(client.server_name().as_deref(), Some("test-server"));
    assert!(client.server_capabilities().tools.is_some());
    assert!(!client.protocol_version().is_empty());
    let stats = exercise_session(&client).await;
    assert_eq!(stats["initialize"], json!(1), "{stats}");
    assert_eq!(stats["tools_list"], json!(2 * tools_pages()), "{stats}");
    client.close().await.expect("close");
}

async fn http_discovery_and_handshake() {
    let server = TestServer::default();
    let counters = server.counters();
    let http = spawn_http(server).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    assert_eq!(client.server_name().as_deref(), Some("test-server"));
    let stats = exercise_session(&client).await;
    assert_eq!(stats["initialize"], json!(1), "{stats}");
    assert_eq!(stats["tools_list"], json!(2 * tools_pages()), "{stats}");
    assert_eq!(
        counters
            .initialize
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    client.close().await.expect("close");
    http.shutdown().await;
}

async fn http_endpoint_query_preserved() {
    let http = spawn_http(TestServer::default()).await;
    let endpoint = format!("{}/?token=abc", http.base);
    let client = McpClient::connect_endpoint(&endpoint, McpTransportType::HttpStream)
        .await
        .expect("connect http endpoint");
    client.list_tools().await.expect("list_tools");
    assert_eq!(
        http.last_query.lock().unwrap().as_deref(),
        Some("token=abc")
    );
    client.close().await.expect("close");
    http.shutdown().await;
}

async fn capability_gating_tools_only() {
    use std::sync::atomic::Ordering;

    let server = TestServer::tools_only();
    let counters = server.counters();
    let http = spawn_http(server).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    assert!(client.server_capabilities().resources.is_none());
    assert!(client.list_resources().await.expect("resources").is_empty());
    assert!(client.list_prompts().await.expect("prompts").is_empty());
    assert_eq!(
        client.list_tools().await.expect("tools").len(),
        all_tools().len()
    );
    assert_eq!(counters.resources_list.load(Ordering::SeqCst), 0);
    assert_eq!(counters.prompts_list.load(Ordering::SeqCst), 0);
    client.close().await.expect("close");
    http.shutdown().await;

    let client = McpClient::connect(stdio_transport(STDIO_TOOLS_ONLY_FLAG))
        .await
        .expect("connect stdio");
    assert!(client.list_resources().await.expect("resources").is_empty());
    assert!(client.list_prompts().await.expect("prompts").is_empty());
    let stats = client.call_tool("stats", Map::new()).await.expect("stats");
    assert_eq!(stats["resources_list"], json!(0), "{stats}");
    assert_eq!(stats["prompts_list"], json!(0), "{stats}");
    client.close().await.expect("close");
}

async fn repeated_cursor_errors() {
    let http = spawn_http(TestServer::repeating_cursor()).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    assert_err_contains(client.list_tools().await, "repeated cursor");
    assert_err_contains(client.list_tools().await, "tools/list");
    client.close().await.expect("close");
    http.shutdown().await;
}

async fn http_connect_error_redacts_query() {
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr").port()
    };
    let url = format!("http://user:pw@127.0.0.1:{port}/mcp?token=secret123");
    let err = McpClient::connect(McpTransport::streamable_http(url))
        .await
        .expect_err("nothing is listening");
    let text = err.to_string();
    assert!(text.contains("failed to connect"), "{text}");
    for leaked in ["secret123", "pw"] {
        assert!(!text.contains(leaked), "{leaked} leaked: {text}");
    }
}

/// rmcp logs transport errors through `tracing` itself; a reqwest error's
/// Display carries the request URL, so those lines must not leak the query.
async fn http_transport_logs_redact_query() {
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr").port()
    };
    let marker = support::logs::mark();
    let url = format!("http://127.0.0.1:{port}/mcp?token=secret123");
    McpClient::connect(McpTransport::streamable_http(url))
        .await
        .expect_err("nothing is listening");
    // The worker logs its quit reason from its own task; give it a moment.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let logs = support::logs::since(marker);
    // The failure must actually have been logged by rmcp, with the error.
    assert!(
        logs.contains("rmcp::transport") && logs.contains("Connection refused"),
        "no rmcp transport error captured: {logs}"
    );
    assert!(
        !logs.contains("secret123"),
        "query leaked into logs: {logs}"
    );
}

async fn closed_session_errors() {
    let http = spawn_http(TestServer::default()).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    let clone = client.clone();
    client.close().await.expect("close");
    client.close().await.expect("second close is a no-op");
    assert_err_contains(client.list_tools().await, "closed");
    assert_err_contains(clone.call_tool("echo", Map::new()).await, "closed");
    http.shutdown().await;
}

type BoxedTool = Box<dyn Tool<Input = McpToolInput, Output = McpToolOutput>>;

fn find_tool(tools: &[McpTool], name: &str) -> McpTool {
    tools
        .iter()
        .find(|t| t.mcp_name() == name)
        .unwrap_or_else(|| panic!("tool {name} not discovered"))
        .clone()
}

/// Exercises `Tool::run` (which blocks, so it needs this multi-thread
/// runtime) and `McpTool::call` against a live session.
async fn assert_tool_trait(client: &McpClient) {
    let tools = client.tools().await.expect("tools");
    assert_eq!(tools.len(), all_tools().len());

    let add = find_tool(&tools, "add");
    assert!(add.output_schema().is_some());
    assert_eq!(add.input_schema()["type"], json!("object"));
    let boxed: BoxedTool = Box::new(add.clone());
    assert_eq!(boxed.name(), "add");
    assert!(!boxed.description().is_empty());
    let out = boxed
        .run(McpToolInput(args(json!({"a": 2, "b": 3}))))
        .expect("run add");
    assert_eq!(out.result["sum"].as_f64(), Some(5.0), "{}", out.result);

    let boom = find_tool(&tools, "boom");
    assert_err_contains(boom.call(Map::new()).await, "it broke");
    let boxed_boom: BoxedTool = Box::new(boom);
    assert_err_contains(boxed_boom.run(McpToolInput::default()), "it broke");
    let echo = find_tool(&tools, "echo");
    assert_eq!(
        echo.call(args(json!({"text": "hi"}))).await.expect("echo"),
        json!("hi")
    );
    let json_text = find_tool(&tools, "json_text");
    assert_eq!(
        json_text.call(Map::new()).await.expect("json_text"),
        json!({"k": 1})
    );
}

async fn tool_trait_over_http() {
    let http = spawn_http(TestServer::default()).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    assert_tool_trait(&client).await;
    client.close().await.expect("close");
    http.shutdown().await;
}

async fn tool_trait_over_stdio() {
    let client = McpClient::connect(stdio_transport(STDIO_FLAG))
        .await
        .expect("connect stdio");
    assert_tool_trait(&client).await;
    client.close().await.expect("close");
}

/// `Tool::run` must return an error, not panic, on a current-thread runtime
/// and outside any runtime.
async fn tool_run_on_current_thread_runtime_errors() {
    let client = McpClient::connect(stdio_transport(STDIO_FLAG))
        .await
        .expect("connect stdio");
    let tool = find_tool(&client.tools().await.expect("tools"), "echo");
    let for_thread = tool.clone();
    std::thread::spawn(move || {
        let boxed: BoxedTool = Box::new(for_thread);
        // Outside any runtime.
        assert_err_contains(boxed.run(McpToolInput::default()), "multi-threaded");
        // On a current-thread runtime.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            assert_err_contains(boxed.run(McpToolInput::default()), "multi-threaded");
        });
    })
    .join()
    .expect("run must not panic");
    client.close().await.expect("close");
}

type BoxedResource = Box<dyn Resource<Input = McpToolInput, Output = McpResourceOutput>>;

fn find_resource(resources: &[McpResource], uri: &str) -> McpResource {
    resources
        .iter()
        .find(|r| r.definition().uri == uri)
        .unwrap_or_else(|| panic!("resource {uri} not discovered"))
        .clone()
}

/// Discovery, reads (blocking and async), template filling and an unknown URI.
/// `Resource::read` blocks, so this must run on a multi-thread runtime.
async fn assert_resources(client: &McpClient) {
    let resources = client.resources().await.expect("resources");
    let uris: Vec<&str> = resources
        .iter()
        .map(|r| r.definition().uri.as_str())
        .collect();
    assert_eq!(uris, ["mem://hello", "mem://blob", "mem://notes/{id}"]);
    let templates: Vec<bool> = resources
        .iter()
        .map(|r| r.definition().is_template)
        .collect();
    assert_eq!(templates, [false, false, true]);

    let hello = client.read_resource("mem://hello").await.expect("hello");
    assert_eq!(hello.mime_type.as_deref(), Some("text/plain"));
    assert_eq!(
        hello.contents,
        [McpResourceContent::Text {
            uri: "mem://hello".into(),
            mime_type: Some("text/plain".into()),
            text: "hi".into(),
        }]
    );
    let blob = find_resource(&resources, "mem://blob")
        .read_async(Map::new())
        .await
        .expect("blob");
    assert_eq!(
        blob.contents,
        [McpResourceContent::Blob {
            uri: "mem://blob".into(),
            mime_type: Some("application/octet-stream".into()),
            blob: "AAEC".into(),
        }]
    );

    let note = find_resource(&resources, "mem://notes/{id}");
    assert_eq!(
        note.uri_for(&args(json!({"id": 42}))).unwrap(),
        "mem://notes/42"
    );
    let boxed: BoxedResource = Box::new(note.clone());
    assert_eq!(boxed.name(), "note");
    let out = boxed
        .read(McpToolInput(args(json!({"id": "42"}))))
        .expect("read template");
    assert!(
        matches!(&out.contents[..], [McpResourceContent::Text { text, .. }] if text == "note 42"),
        "{out:?}"
    );
    assert_err_contains(boxed.read(McpToolInput::default()), "'id'");

    assert!(client.read_resource("mem://nope").await.is_err());
    let blob_res: BoxedResource = Box::new(find_resource(&resources, "mem://blob"));
    let again = blob_res
        .read(McpToolInput::default())
        .expect("blob via trait");
    assert_eq!(again, blob);
}

async fn resources_over_http() {
    let http = spawn_http(TestServer::default()).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    assert_resources(&client).await;
    client.close().await.expect("close");
    http.shutdown().await;
}

async fn resources_over_stdio() {
    let client = McpClient::connect(stdio_transport(STDIO_FLAG))
        .await
        .expect("connect stdio");
    assert_resources(&client).await;
    client.close().await.expect("close");
}

/// A server without `resources/templates/list` still lists its concrete
/// resources, with no error.
async fn resource_templates_method_not_found_tolerated() {
    let http = spawn_http(TestServer::without_templates()).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    let defs = client.list_resources().await.expect("list_resources");
    let uris: Vec<&str> = defs.iter().map(|d| d.uri.as_str()).collect();
    assert_eq!(uris, ["mem://hello", "mem://blob"]);
    assert!(defs.iter().all(|d| !d.is_template));
    client.close().await.expect("close");
    http.shutdown().await;
}

/// `Resource::read` must return an error, not panic, on a current-thread
/// runtime and outside any runtime.
async fn resource_read_on_current_thread_runtime_errors() {
    let client = McpClient::connect(stdio_transport(STDIO_FLAG))
        .await
        .expect("connect stdio");
    let hello = find_resource(&client.resources().await.expect("resources"), "mem://hello");
    std::thread::spawn(move || {
        let boxed: BoxedResource = Box::new(hello);
        assert_err_contains(boxed.read(McpToolInput::default()), "multi-threaded");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            assert_err_contains(
                boxed.read(McpToolInput::default()),
                "McpResource::read_async",
            );
        });
    })
    .join()
    .expect("read must not panic");
    client.close().await.expect("close");
}

/// The server echoes the requested URI in its error, and a caller-supplied
/// URI may carry the endpoint's secret; `resources/read` errors must be
/// scrubbed like every other MCP error.
async fn resource_read_error_redacts_query() {
    let http = spawn_http(TestServer::default()).await;
    let endpoint = format!("{}/mcp?token=secret123", http.base);
    let client = McpClient::connect_endpoint(&endpoint, McpTransportType::HttpStream)
        .await
        .expect("connect http endpoint");
    let err = client
        .read_resource("mem://nope?token=secret123")
        .await
        .expect_err("unknown resource");
    let text = err.to_string();
    assert!(text.contains("resources/read"), "{text}");
    assert!(!text.contains("secret123"), "secret leaked: {text}");
    client.close().await.expect("close");
    http.shutdown().await;
}

type BoxedPrompt = Box<dyn Prompt<Input = McpToolInput, Output = McpPromptOutput>>;

fn find_prompt(prompts: &[McpPrompt], name: &str) -> McpPrompt {
    prompts
        .iter()
        .find(|p| p.definition().name == name)
        .unwrap_or_else(|| panic!("prompt {name} not discovered"))
        .clone()
}

/// Discovery, blocking and async generation, and a missing required argument.
/// `Prompt::generate` blocks, so this must run on a multi-thread runtime.
async fn assert_prompts(client: &McpClient) {
    let prompts = client.prompts().await.expect("prompts");
    let greet = find_prompt(&prompts, "greet");
    let args_def = &greet.definition().arguments;
    assert_eq!(args_def.len(), 1);
    assert_eq!(args_def[0].name, "name");
    assert!(args_def[0].required);

    let boxed: BoxedPrompt = Box::new(greet.clone());
    assert_eq!(boxed.name(), "greet");
    let out = boxed
        .generate(McpToolInput(args(json!({"name": "Ada"}))))
        .expect("generate");
    assert_eq!(out.content, "Hello, Ada!");
    assert_eq!(out.description.as_deref(), Some("Greets someone"));
    assert_eq!(out.messages[0].role, "user");
    assert_eq!(
        out.messages[0].content,
        json!({"type": "text", "text": "Hello, Ada!"})
    );

    let via_client = client
        .get_prompt("greet", args(json!({"name": "Ada"})))
        .await
        .expect("get_prompt");
    assert_eq!(via_client, out);
    let via_async = greet.get(args(json!({"name": "Ada"}))).await.expect("get");
    assert_eq!(via_async, out);

    let missing = boxed
        .generate(McpToolInput::default())
        .expect_err("missing argument");
    let text = missing.to_string();
    assert!(
        text.contains("prompts/get") && text.contains("greet"),
        "{text}"
    );
    let unknown = client
        .get_prompt("nope", Map::new())
        .await
        .expect_err("unknown prompt");
    let text = unknown.to_string();
    assert!(
        text.contains("prompts/get") && text.contains("nope"),
        "{text}"
    );
}

async fn prompts_over_http() {
    let http = spawn_http(TestServer::default()).await;
    let client = McpClient::connect(McpTransport::streamable_http(format!("{}/mcp", http.base)))
        .await
        .expect("connect http");
    assert_prompts(&client).await;
    client.close().await.expect("close");
    http.shutdown().await;
}

async fn prompts_over_stdio() {
    let client = McpClient::connect(stdio_transport(STDIO_FLAG))
        .await
        .expect("connect stdio");
    assert_prompts(&client).await;
    client.close().await.expect("close");
}

/// `Prompt::generate` must return an error, not panic, on a current-thread
/// runtime and outside any runtime.
async fn prompt_generate_on_current_thread_runtime_errors() {
    let client = McpClient::connect(stdio_transport(STDIO_FLAG))
        .await
        .expect("connect stdio");
    let greet = find_prompt(&client.prompts().await.expect("prompts"), "greet");
    std::thread::spawn(move || {
        let boxed: BoxedPrompt = Box::new(greet);
        assert_err_contains(boxed.generate(McpToolInput::default()), "multi-threaded");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            assert_err_contains(boxed.generate(McpToolInput::default()), "McpPrompt::get");
        });
    })
    .join()
    .expect("generate must not panic");
    client.close().await.expect("close");
}

/// The server echoes the requested prompt name in its error, and a caller
/// may pass a name carrying the endpoint's secret; `prompts/get` errors must
/// be scrubbed like every other MCP error.
async fn prompt_get_error_redacts_query() {
    let http = spawn_http(TestServer::default()).await;
    let endpoint = format!("{}/mcp?token=secret123", http.base);
    let client = McpClient::connect_endpoint(&endpoint, McpTransportType::HttpStream)
        .await
        .expect("connect http endpoint");
    let err = client
        .get_prompt("nope?token=secret123", Map::new())
        .await
        .expect_err("unknown prompt");
    let text = err.to_string();
    assert!(text.contains("prompts/get"), "{text}");
    assert!(!text.contains("secret123"), "secret leaked: {text}");
    client.close().await.expect("close");
    http.shutdown().await;
}

/// Legacy HTTP+SSE: endpoint resolution, the 2024-11-05 handshake, paged
/// discovery, calls, resources and prompts over one session.
async fn sse_discovery_calls_resources_prompts() {
    use std::sync::atomic::Ordering;

    let server = TestServer::default();
    let counters = server.counters();
    let legacy = spawn_legacy_sse(server, LegacyOptions::default()).await;
    let client = McpClient::connect_endpoint(&legacy.base, McpTransportType::Sse)
        .await
        .expect("connect sse");
    assert_eq!(client.protocol_version(), "2024-11-05");
    assert_discovery(&client).await;
    let echo = client.call_tool("echo", args(json!({"text": "hi"}))).await;
    assert_eq!(echo.expect("echo"), json!("hi"));
    let hello = client.read_resource("mem://hello").await.expect("read");
    assert!(!hello.contents.is_empty(), "{hello:?}");
    let greet = client
        .get_prompt("greet", args(json!({"name": "Ada"})))
        .await
        .expect("get_prompt");
    assert_eq!(greet.content, "Hello, Ada!");
    assert_eq!(counters.initialize.load(Ordering::SeqCst), 1);
    assert_eq!(
        counters.tools_list.load(Ordering::SeqCst),
        tools_pages(),
        "tools/list pages"
    );
    client.close().await.expect("close");
    legacy.shutdown().await;
}

/// An absolute endpoint on another origin must be refused, not posted to.
async fn sse_cross_origin_endpoint_rejected() {
    let legacy = spawn_legacy_sse(
        TestServer::default(),
        LegacyOptions {
            endpoint_override: Some("http://evil.example/messages?session=1".into()),
            ..LegacyOptions::default()
        },
    )
    .await;
    let err = match McpClient::connect_endpoint(&legacy.base, McpTransportType::Sse).await {
        Ok(_) => panic!("cross-origin endpoint accepted"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("same-origin"), "{err}");
    assert!(!err.contains("evil.example"), "endpoint leaked: {err}");
    legacy.shutdown().await;
}

/// Configured headers ride on the SSE `GET` and on every message `POST`; the
/// server rejects requests without them.
async fn sse_header_forwarded() {
    use std::sync::atomic::Ordering;

    let legacy = spawn_legacy_sse(
        TestServer::default(),
        LegacyOptions {
            required_header: Some(("x-api-key".into(), "k".into())),
            ..LegacyOptions::default()
        },
    )
    .await;
    let transport = McpTransport::sse(format!("{}/sse", legacy.base)).with_header("x-api-key", "k");
    let client = McpClient::connect(transport).await.expect("connect sse");
    client.list_tools().await.expect("list_tools");
    assert!(legacy.observed.get_had_header.load(Ordering::SeqCst));
    assert!(legacy.observed.posts.load(Ordering::SeqCst) >= 3);
    assert!(legacy.observed.post_had_header.load(Ordering::SeqCst));
    client.close().await.expect("close");

    let bare = McpClient::connect(McpTransport::sse(format!("{}/sse", legacy.base))).await;
    assert!(bare.is_err(), "connect without the header must fail");
    legacy.shutdown().await;
}

/// Dropping a `connect` that is still waiting for the `endpoint` event must
/// close the SSE connection instead of leaking the reader task.
async fn sse_cancelled_connect_closes_stream() {
    use std::sync::atomic::Ordering;

    let legacy = spawn_legacy_sse(
        TestServer::default(),
        LegacyOptions {
            never_send_endpoint: true,
            ..LegacyOptions::default()
        },
    )
    .await;
    let connect = McpClient::connect(McpTransport::sse(format!("{}/sse", legacy.base)));
    let outcome = tokio::time::timeout(Duration::from_millis(300), connect).await;
    assert!(outcome.is_err(), "connect must still be waiting");
    assert_eq!(legacy.observed.streams_opened.load(Ordering::SeqCst), 1);
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while legacy.observed.live_streams.load(Ordering::SeqCst) != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "SSE connection still open after connect was cancelled"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    legacy.shutdown().await;
}

/// Discovers the HTTP server's tools, minus `stats`, as an orchestrator toolset.
async fn orchestrator_toolset(base: &str) -> (McpClient, McpToolset) {
    let (client, tools, schema) =
        fetch_mcp_attributes_with_schema(&format!("{base}/mcp"), McpTransportType::HttpStream)
            .await
            .expect("fetch tools and schema");
    assert_eq!(
        schema["properties"]["tool_parameters"]["anyOf"]
            .as_array()
            .map(Vec::len),
        Some(tools.len())
    );
    let tools = tools
        .into_iter()
        .filter(|t| t.mcp_name() != "stats")
        .collect();
    (client, McpToolset::new(tools).expect("toolset"))
}

async fn orchestrator_selects_mcp_tool() {
    let http = spawn_http(TestServer::default()).await;
    let (client, toolset) = orchestrator_toolset(&http.base).await;

    let (llm, requests) =
        ScriptedLlm::new([json!({"tool_parameters": {"tool_name": "add", "a": 2, "b": 3}})]);
    let agent = AtomicAgent::<BasicChatInputSchema, McpOrchestratorOutput, _>::new(
        AgentConfig::new(llm, "test-model").with_output_schema(toolset.schema()),
    );
    let out = agent
        .run_async(Some(BasicChatInputSchema {
            chat_message: "add 2 and 3".into(),
        }))
        .await
        .expect("agent run");
    assert_eq!(out.tool_parameters.tool_name, "add");
    assert_eq!(
        Value::Object(out.tool_parameters.arguments.clone()),
        json!({"a": 2, "b": 3})
    );

    let result = toolset
        .dispatch(&out.tool_parameters)
        .await
        .expect("dispatch");
    assert_eq!(result["sum"].as_f64(), Some(5.0), "{result}");

    {
        let recorded = requests.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let tools = recorded[0].tools.as_ref().expect("tools in request");
        assert_eq!(tools[0].function.parameters, toolset.schema());
    }
    let names: Vec<_> = toolset.tools().iter().map(|t| t.mcp_name()).collect();
    assert!(!names.contains(&"stats"));
    client.close().await.expect("close");
    http.shutdown().await;
}

async fn orchestrator_unknown_tool_dispatch_errors() {
    let http = spawn_http(TestServer::default()).await;
    let (client, toolset) = orchestrator_toolset(&http.base).await;
    assert!(toolset.get("add").is_some());
    assert!(toolset.get("stats").is_none());

    for name in ["nope", "stats"] {
        let call = McpToolCall {
            tool_name: name.to_string(),
            arguments: Map::new(),
        };
        let available = toolset
            .tools()
            .iter()
            .map(|t| t.mcp_name())
            .collect::<Vec<_>>()
            .join(", ");
        assert!(available.contains("add") && !available.contains("stats"));
        assert_err_contains(
            toolset.dispatch(&call).await,
            &format!("unknown MCP tool {name:?}; available: {available}"),
        );
    }
    client.close().await.expect("close");
    http.shutdown().await;
}

/// The compat service opens a real session per call and closes it again.
async fn definition_service_one_shot() {
    // `from_endpoint` shlex-splits the endpoint, so the binary path is quoted.
    let exe = std::env::current_exe().expect("current exe");
    let quoted = shlex::try_quote(&exe.to_string_lossy())
        .expect("path has no NUL")
        .into_owned();
    let service =
        McpDefinitionService::new(format!("{quoted} {STDIO_FLAG}"), McpTransportType::Stdio);

    let tools = service.fetch_tool_definitions().await.expect("tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    let expected = all_tools();
    let expected_names: Vec<&str> = expected.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(names, expected_names);
    assert_eq!(tools.len(), 5);

    let resources = service
        .fetch_resource_definitions()
        .await
        .expect("resources");
    let uris: Vec<&str> = resources.iter().map(|r| r.uri.as_str()).collect();
    assert_eq!(uris, ["mem://hello", "mem://blob", "mem://notes/{id}"]);
    let prompts = service.fetch_prompt_definitions().await.expect("prompts");
    assert!(prompts.iter().any(|p| p.name == "greet"));

    let service = service.with_working_directory(std::env::temp_dir());
    assert_eq!(
        service.fetch_tool_definitions().await.expect("tools").len(),
        5
    );
}
