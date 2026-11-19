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

use atomic_agents::error::Error;
use atomic_agents::mcp::{McpClient, McpTransport, McpTransportType};
use futures::FutureExt;
use futures::future::BoxFuture;
use rmcp::ServiceExt;
use serde_json::{Map, Value, json};
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
        ("closed_session_errors", || closed_session_errors().boxed()),
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
            "--ignored" | "--include-ignored" => ignored = true,
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
