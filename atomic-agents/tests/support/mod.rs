//! Shared fixtures for the MCP integration harness.

pub mod legacy_sse_server;
pub mod logs;
pub mod scripted_llm;
pub mod server;

use std::sync::{Arc, Mutex};

use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub use server::TestServer;

/// A Streamable HTTP MCP server listening on a loopback port.
pub struct HttpServer {
    /// Base URL without a path, e.g. `http://127.0.0.1:40123`.
    pub base: String,
    /// Query string of the most recent request (`None` when it had none).
    pub last_query: Arc<Mutex<Option<String>>>,
    ct: CancellationToken,
    handle: JoinHandle<()>,
}

impl HttpServer {
    /// Stops accepting connections and waits for the server task to end.
    pub async fn shutdown(self) {
        self.ct.cancel();
        let _ = self.handle.await;
    }
}

/// Serves `server` over Streamable HTTP at `/mcp` on `127.0.0.1:0`.
pub async fn spawn_http(server: TestServer) -> HttpServer {
    let ct = CancellationToken::new();
    let service: StreamableHttpService<TestServer, LocalSessionManager> =
        StreamableHttpService::new(
            move || Ok(server.clone()),
            Default::default(),
            StreamableHttpServerConfig::default().with_cancellation_token(ct.child_token()),
        );
    let last_query: Arc<Mutex<Option<String>>> = Arc::default();
    let recorder = last_query.clone();
    let router =
        axum::Router::new()
            .nest_service("/mcp", service)
            .layer(axum::middleware::from_fn(
                move |req: axum::extract::Request, next: axum::middleware::Next| {
                    *recorder.lock().unwrap() = req.uri().query().map(str::to_string);
                    next.run(req)
                },
            ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener address");
    let handle = tokio::spawn({
        let ct = ct.clone();
        async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async move { ct.cancelled_owned().await })
                .await
                .expect("axum server");
        }
    });
    HttpServer {
        base: format!("http://{addr}"),
        last_query,
        ct,
        handle,
    }
}
