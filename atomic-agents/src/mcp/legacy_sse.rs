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
use std::time::Duration;

use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::header::HeaderMap;
use rmcp::RoleClient;
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::Transport;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// How long to wait for the server's `endpoint` event.
const ENDPOINT_TIMEOUT: Duration = Duration::from_secs(30);

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

pub(crate) struct LegacySseTransport {
    http: reqwest::Client,
    headers: HeaderMap,
    post_url: url::Url,
    rx: mpsc::Receiver<ServerJsonRpcMessage>,
    reader: JoinHandle<()>,
}

impl LegacySseTransport {
    /// Opens the SSE stream at `sse_url`, waits for the `endpoint` event and
    /// returns a transport that posts to it. `headers` are sent on the `GET`
    /// and on every `POST`. An endpoint on a different origin is rejected so
    /// that credentials are never replayed to a third party.
    pub(crate) async fn connect(
        http: reqwest::Client,
        sse_url: &str,
        headers: HeaderMap,
    ) -> Result<Self, LegacySseError> {
        let sse_url = url::Url::parse(sse_url)
            .map_err(|e| LegacySseError(format!("invalid SSE URL: {e}")))?;
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

        let (tx, rx) = mpsc::channel(64);
        let (ep_tx, ep_rx) = oneshot::channel::<String>();
        let mut ep_tx = Some(ep_tx);
        let reader = tokio::spawn(async move {
            let mut events = resp.bytes_stream().eventsource();
            while let Some(next) = events.next().await {
                let event = match next {
                    Ok(event) => event,
                    Err(_) => {
                        // The error may embed a URL; log only that it happened.
                        tracing::warn!("legacy SSE stream error; closing transport");
                        return;
                    }
                };
                match event.event.as_str() {
                    "endpoint" => {
                        if let Some(t) = ep_tx.take() {
                            let _ = t.send(event.data);
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
            // Dropping `tx` makes `receive` yield `None`, which rmcp treats as
            // the transport closing.
        });

        let endpoint = match tokio::time::timeout(ENDPOINT_TIMEOUT, ep_rx).await {
            Ok(Ok(endpoint)) => endpoint,
            Ok(Err(_)) => {
                return Err(LegacySseError("stream ended before endpoint event".into()));
            }
            Err(_) => {
                reader.abort();
                return Err(LegacySseError("timeout waiting for endpoint event".into()));
            }
        };
        let post_url = match sse_url.join(&endpoint) {
            Ok(u) => u,
            Err(_) => {
                reader.abort();
                return Err(LegacySseError("invalid endpoint URL".into()));
            }
        };
        if post_url.origin() != sse_url.origin() {
            reader.abort();
            return Err(LegacySseError(
                "legacy SSE endpoint must be same-origin".into(),
            ));
        }
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
        self.reader.abort();
        Ok(())
    }
}

impl Drop for LegacySseTransport {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
