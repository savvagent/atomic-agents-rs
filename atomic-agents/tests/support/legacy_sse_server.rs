//! Legacy HTTP+SSE (2024-11-05) MCP server for the integration harness.
//!
//! `GET /sse` opens a session: it sends an `endpoint` event naming
//! `/messages?session=<n>` and then forwards the session's server-to-client
//! messages as `message` events. `POST /messages?session=<n>` hands the body
//! to that session's rmcp server (running over an in-memory duplex pipe) and
//! answers 202.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use futures::{Stream, StreamExt};
use rmcp::ServiceExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, WriteHalf};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::TestServer;

type Writers = Arc<Mutex<HashMap<String, WriteHalf<tokio::io::DuplexStream>>>>;

/// Observations about the requests the legacy server received.
#[derive(Debug, Default)]
pub struct LegacyObserved {
    /// The `GET /sse` request carried the required header.
    pub get_had_header: AtomicBool,
    /// Every `POST /messages` request carried the required header.
    pub post_had_header: AtomicBool,
    /// Number of `POST /messages` requests received.
    pub posts: AtomicUsize,
}

#[derive(Clone)]
struct AppState {
    server: TestServer,
    writers: Writers,
    /// Value of the endpoint event's data.
    endpoint_override: Option<String>,
    /// Header (`name`, `value`) every request must carry, when set.
    required_header: Option<(String, String)>,
    observed: Arc<LegacyObserved>,
}

impl AppState {
    fn has_required_header(&self, headers: &HeaderMap) -> bool {
        match &self.required_header {
            None => true,
            Some((n, v)) => headers.get(n.as_str()).is_some_and(|h| h == v.as_str()),
        }
    }
}

/// A running legacy SSE server on a loopback port.
pub struct LegacyServer {
    /// Base URL without a path, e.g. `http://127.0.0.1:40123`.
    pub base: String,
    /// What the server saw.
    pub observed: Arc<LegacyObserved>,
    ct: CancellationToken,
    handle: JoinHandle<()>,
}

impl LegacyServer {
    /// Stops accepting connections and waits for the server task to end.
    pub async fn shutdown(self) {
        self.ct.cancel();
        let _ = self.handle.await;
    }
}

/// Configuration for [`spawn_legacy_sse`].
#[derive(Default)]
pub struct LegacyOptions {
    /// Send this as the `endpoint` event's data instead of `/messages?session=<n>`.
    pub endpoint_override: Option<String>,
    /// Require this header on every request (401 otherwise).
    pub required_header: Option<(String, String)>,
}

async fn sse(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    if state.has_required_header(&headers) {
        state.observed.get_had_header.store(true, Ordering::SeqCst);
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let (client_side, server_side) = tokio::io::duplex(64 * 1024);
    let server = state.server.clone();
    tokio::spawn(async move {
        if let Ok(running) = server.serve(server_side).await {
            let _ = running.waiting().await;
        }
    });
    let (read, write) = tokio::io::split(client_side);
    let id = {
        let mut writers = state.writers.lock().await;
        let id = (writers.len() + 1).to_string();
        writers.insert(id.clone(), write);
        id
    };
    let endpoint = state
        .endpoint_override
        .clone()
        .unwrap_or_else(|| format!("/messages?session={id}"));
    let first = futures::stream::once(async move {
        Ok::<_, Infallible>(Event::default().event("endpoint").data(endpoint))
    });
    let rest = futures::stream::unfold(BufReader::new(read).lines(), |mut lines| async move {
        lines
            .next_line()
            .await
            .ok()
            .flatten()
            .map(|line| (Ok(Event::default().event("message").data(line)), lines))
    });
    Ok(Sse::new(first.chain(rest)))
}

async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    body: String,
) -> StatusCode {
    state.observed.posts.fetch_add(1, Ordering::SeqCst);
    if !state.has_required_header(&headers) {
        state
            .observed
            .post_had_header
            .store(false, Ordering::SeqCst);
        return StatusCode::UNAUTHORIZED;
    }
    let mut writers = state.writers.lock().await;
    let Some(writer) = query.get("session").and_then(|s| writers.get_mut(s)) else {
        return StatusCode::NOT_FOUND;
    };
    if writer.write_all(body.as_bytes()).await.is_err() || writer.write_all(b"\n").await.is_err() {
        return StatusCode::GONE;
    }
    StatusCode::ACCEPTED
}

/// Serves `server` over legacy HTTP+SSE at `/sse` on `127.0.0.1:0`.
pub async fn spawn_legacy_sse(server: TestServer, options: LegacyOptions) -> LegacyServer {
    let observed = Arc::new(LegacyObserved::default());
    // Starts true; any POST without the header flips it to false.
    observed.post_had_header.store(true, Ordering::SeqCst);
    let state = AppState {
        server,
        writers: Writers::default(),
        endpoint_override: options.endpoint_override,
        required_header: options.required_header,
        observed: observed.clone(),
    };
    let router = axum::Router::new()
        .route("/sse", get(sse))
        .route("/messages", post(messages))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener address");
    let ct = CancellationToken::new();
    let handle = tokio::spawn({
        let ct = ct.clone();
        async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async move { ct.cancelled_owned().await })
                .await
                .expect("axum server");
        }
    });
    LegacyServer {
        base: format!("http://{addr}"),
        observed,
        ct,
        handle,
    }
}
