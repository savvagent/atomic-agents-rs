//! Legacy HTTP+SSE (protocol 2024-11-05) client transport.
//!
//! The client opens a long-lived `GET` on the SSE URL. The server first sends
//! an `endpoint` event naming the URL to `POST` client messages to; every
//! server message then arrives as a `message` event on the same stream.
//!
//! Error text produced here never contains a URL: the SSE URL and the
//! server-chosen POST endpoint both commonly carry credentials or session ids
//! in their query, and `reqwest` errors normally embed the request URL, so
//! they are stripped with `without_url` before formatting.

use std::fmt;
use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use eventsource_stream::{EventStreamError, Eventsource};
use futures::{Stream, StreamExt};
use reqwest::header::HeaderMap;
use rmcp::RoleClient;
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::Transport;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// Default time allowed for the whole connect sequence: the `GET`, its status
/// check, and the wait for the server's `endpoint` event.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default cap on bytes buffered without an event boundary. A server that
/// never terminates an event would otherwise grow the parser buffer without
/// bound.
const MAX_SSE_EVENT_BYTES: usize = 16 * 1024 * 1024;

/// Tunable limits; tests lower them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    /// Total time allowed for [`LegacySseTransport::connect_with`].
    pub(crate) connect_timeout: Duration,
    /// Most bytes accepted without an event boundary.
    pub(crate) max_event_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            connect_timeout: CONNECT_TIMEOUT,
            max_event_bytes: MAX_SSE_EVENT_BYTES,
        }
    }
}

/// Failure in the legacy SSE transport. The text never includes a URL.
#[derive(Debug)]
pub(crate) struct LegacySseError(pub(crate) String);

impl fmt::Display for LegacySseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "legacy SSE: {}", self.0)
    }
}

impl std::error::Error for LegacySseError {}

/// Describes a `reqwest` error without the request URL.
fn describe(e: reqwest::Error) -> String {
    e.without_url().to_string()
}

/// Aborts the wrapped task when dropped, so a cancelled `connect` (or a
/// dropped transport) cannot leave the reader task holding the connection.
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Resolves the `endpoint` event's `data` against the SSE URL.
///
/// The endpoint must be on the SSE URL's origin. The configured headers
/// (often credentials) are sent on every `POST`, so honouring an endpoint on
/// another origin would replay them to a third party. The error text never
/// echoes the server-supplied value, which may carry a session id.
fn resolve_post_url(sse: &url::Url, data: &str) -> Result<url::Url, LegacySseError> {
    let data = data.trim();
    if data.is_empty() {
        return Err(LegacySseError("empty endpoint".into()));
    }
    let post = sse
        .join(data)
        .map_err(|_| LegacySseError("invalid endpoint URL".into()))?;
    if post.origin() != sse.origin() {
        return Err(LegacySseError(
            "legacy SSE endpoint must be same-origin".into(),
        ));
    }
    Ok(post)
}

/// Ends `stream` (as if the connection closed) once more than `max` bytes
/// arrive without an event boundary (`\n\n` or `\r\n\r\n`); the count resets at
/// each boundary, including one split across chunks.
fn cap_event_bytes<S, E>(stream: S, max: usize) -> impl Stream<Item = Result<Bytes, E>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    struct State<S> {
        stream: Pin<Box<S>>,
        since_boundary: usize,
        /// Last up-to-3 bytes, so a boundary split across chunks is seen.
        tail: Vec<u8>,
    }
    fn last_boundary_end(buf: &[u8]) -> Option<usize> {
        let find = |pat: &[u8]| {
            buf.windows(pat.len())
                .rposition(|w| w == pat)
                .map(|i| i + pat.len())
        };
        find(b"\n\n").max(find(b"\r\n\r\n"))
    }
    let state = State {
        stream: Box::pin(stream),
        since_boundary: 0,
        tail: Vec::new(),
    };
    futures::stream::unfold(state, move |mut st| async move {
        let item = st.stream.next().await?;
        if let Ok(chunk) = &item {
            let mut window = st.tail.clone();
            window.extend_from_slice(chunk);
            st.since_boundary = match last_boundary_end(&window) {
                Some(end) => window.len() - end,
                None => st.since_boundary + chunk.len(),
            };
            st.tail = window[window.len().saturating_sub(3)..].to_vec();
            if st.since_boundary > max {
                tracing::warn!(max, "legacy SSE event exceeds size cap; closing transport");
                return None;
            }
        }
        Some((item, st))
    })
}

pub(crate) struct LegacySseTransport {
    http: reqwest::Client,
    /// Sent per request (not baked into the client) because the client is
    /// supplied by the caller; the effect is the same: the `GET` and every
    /// `POST` carry them.
    headers: HeaderMap,
    post_url: url::Url,
    rx: mpsc::Receiver<ServerJsonRpcMessage>,
    reader: AbortOnDrop,
}

impl LegacySseTransport {
    /// Opens the SSE stream at `sse_url`, waits for the `endpoint` event and
    /// returns a transport that posts to it. `headers` are sent on the `GET`
    /// and on every `POST`. See [`resolve_post_url`] for the origin rule.
    pub(crate) async fn connect(
        http: reqwest::Client,
        sse_url: &str,
        headers: HeaderMap,
    ) -> Result<Self, LegacySseError> {
        Self::connect_with(http, sse_url, headers, Limits::default()).await
    }

    /// [`connect`](Self::connect) with explicit [`Limits`]. The whole sequence
    /// (`GET`, status check, endpoint wait) is bounded by
    /// `limits.connect_timeout`; on timeout or cancellation the reader task
    /// is aborted.
    pub(crate) async fn connect_with(
        http: reqwest::Client,
        sse_url: &str,
        headers: HeaderMap,
        limits: Limits,
    ) -> Result<Self, LegacySseError> {
        let sse_url = url::Url::parse(sse_url)
            .map_err(|e| LegacySseError(format!("invalid SSE URL: {e}")))?;
        tokio::time::timeout(
            limits.connect_timeout,
            Self::open(http, sse_url, headers, limits.max_event_bytes),
        )
        .await
        .map_err(|_| LegacySseError("timed out connecting (no endpoint event)".into()))?
    }

    async fn open(
        http: reqwest::Client,
        sse_url: url::Url,
        headers: HeaderMap,
        max_event_bytes: usize,
    ) -> Result<Self, LegacySseError> {
        let resp = http
            .get(sse_url.clone())
            .headers(headers.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .send()
            .await
            .map_err(|e| LegacySseError(describe(e)))?;
        if !resp.status().is_success() {
            return Err(LegacySseError(format!("GET {}", resp.status())));
        }

        // Messages that arrive before the endpoint event are queued here
        // (bounded by the channel, which back-pressures the reader) and
        // delivered once the transport is returned.
        let (tx, rx) = mpsc::channel(64);
        let (ep_tx, ep_rx) = oneshot::channel::<Result<String, &'static str>>();
        let mut ep_tx = Some(ep_tx);
        let reader = AbortOnDrop(tokio::spawn(async move {
            let bytes = cap_event_bytes(resp.bytes_stream(), max_event_bytes);
            let mut events = Box::pin(bytes.eventsource());
            // Why the stream ended; only a fixed kind, never a payload.
            let mut ended = "stream closed";
            while let Some(next) = events.next().await {
                let event = match next {
                    Ok(event) => event,
                    Err(e) => {
                        // The error may embed a URL; log only its kind.
                        ended = match e {
                            EventStreamError::Transport(_) => "transport error",
                            _ => "decode error",
                        };
                        tracing::warn!(kind = ended, "legacy SSE stream error; closing transport");
                        break;
                    }
                };
                match event.event.as_str() {
                    "endpoint" => {
                        if let Some(t) = ep_tx.take() {
                            let _ = t.send(Ok(event.data));
                        }
                    }
                    "message" => match serde_json::from_str::<ServerJsonRpcMessage>(&event.data) {
                        Ok(m) => {
                            if tx.send(m).await.is_err() {
                                return;
                            }
                        }
                        Err(_) => tracing::warn!(
                            len = event.data.len(),
                            "skipping malformed legacy SSE message"
                        ),
                    },
                    _ => {}
                }
            }
            if let Some(t) = ep_tx.take() {
                let _ = t.send(Err(ended));
            }
            // Dropping `tx` makes `receive` yield `None`, which rmcp treats as
            // the transport closing.
        }));

        let endpoint = match ep_rx.await {
            Ok(Ok(endpoint)) => endpoint,
            Ok(Err(kind)) => {
                return Err(LegacySseError(format!(
                    "stream ended before endpoint event ({kind})"
                )));
            }
            Err(_) => {
                return Err(LegacySseError(
                    "stream ended before endpoint event (reader stopped)".into(),
                ));
            }
        };
        let post_url = resolve_post_url(&sse_url, &endpoint)?;
        Ok(Self {
            http,
            headers,
            post_url,
            rx,
            reader,
        })
    }
}

impl Transport<RoleClient> for LegacySseTransport {
    type Error = LegacySseError;

    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let (http, url, headers) = (
            self.http.clone(),
            self.post_url.clone(),
            self.headers.clone(),
        );
        async move {
            let resp = http
                .post(url)
                .headers(headers)
                .json(&item)
                .send()
                .await
                .map_err(|e| LegacySseError(describe(e)))?;
            if resp.status().is_success() {
                Ok(())
            } else {
                Err(LegacySseError(format!("POST {}", resp.status())))
            }
        }
    }

    fn receive(&mut self) -> impl Future<Output = Option<ServerJsonRpcMessage>> + Send {
        self.rx.recv()
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.reader.0.abort();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn connect_errors_carry_no_url() {
        let err = match LegacySseTransport::connect(
            reqwest::Client::new(),
            "http://user:pw@127.0.0.1:1/sse?token=secret123",
            HeaderMap::new(),
        )
        .await
        {
            Ok(_) => panic!("nothing listens on port 1"),
            Err(e) => e.to_string(),
        };
        for leaked in ["secret123", "pw", "127.0.0.1"] {
            assert!(!err.contains(leaked), "{leaked} leaked: {err}");
        }
    }

    /// A server that accepts the connection and never answers must not hang
    /// `connect` forever.
    #[tokio::test]
    async fn connect_times_out_when_server_never_responds() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let _held = tokio::spawn(async move {
            let mut sockets = Vec::new();
            while let Ok((s, _)) = listener.accept().await {
                sockets.push(s);
            }
        });
        let limits = Limits {
            connect_timeout: Duration::from_millis(300),
            ..Limits::default()
        };
        let err = match LegacySseTransport::connect_with(
            reqwest::Client::new(),
            &format!("http://{addr}/sse?token=secret123"),
            HeaderMap::new(),
            limits,
        )
        .await
        {
            Ok(_) => panic!("server never responds"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("timed out"), "{err}");
        assert!(!err.contains("secret123"), "{err}");
    }

    #[tokio::test]
    async fn resolve_post_url_table() {
        let sse = url::Url::parse("http://h:8080/a/sse?k=v").unwrap();
        let ok = [
            ("/messages", "http://h:8080/messages"),
            ("/messages?session=7", "http://h:8080/messages?session=7"),
            ("messages?s=1", "http://h:8080/a/messages?s=1"),
            ("  /messages  ", "http://h:8080/messages"),
            ("http://h:8080/m?s=2", "http://h:8080/m?s=2"),
        ];
        for (data, want) in ok {
            assert_eq!(
                resolve_post_url(&sse, data).unwrap().as_str(),
                want,
                "{data}"
            );
        }
        for data in [
            "https://h:8080/m",
            "http://h:9090/m",
            "http://evil.example:8080/m",
            "//evil.example/x",
        ] {
            let err = resolve_post_url(&sse, data).unwrap_err().to_string();
            assert!(err.contains("same-origin"), "{data}: {err}");
            assert!(!err.contains("evil"), "{data}: {err}");
        }
        for data in ["", "   ", "\n\t"] {
            let err = resolve_post_url(&sse, data).unwrap_err().to_string();
            assert!(err.contains("empty endpoint"), "{data:?}: {err}");
        }
    }

    type Chunk = Result<Bytes, std::convert::Infallible>;

    async fn collect_capped(chunks: Vec<&'static str>, cap: usize) -> usize {
        let stream = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok(Bytes::from_static(c.as_bytes())) as Chunk),
        );
        cap_event_bytes(stream, cap).count().await
    }

    #[tokio::test]
    async fn event_size_cap_ends_stream() {
        // Within the cap, boundaries (even split across chunks) reset the count.
        let ok = vec!["data: aaaa\n", "\n", "data: bbbb\r\n\r", "\ndata: cc\n\n"];
        assert_eq!(collect_capped(ok, 14).await, 4);
        // No boundary: the stream ends once the cap is exceeded.
        let big = vec!["data: 1111", "2222", "3333", "4444", "\n\n"];
        assert_eq!(collect_capped(big, 12).await, 1);
    }
}
