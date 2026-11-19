//! A Streamable HTTP client that keeps request URLs out of its errors.
//!
//! rmcp logs transport errors through `tracing` itself (for example the
//! worker's fatal quit reason), and a `reqwest::Error` formats the full
//! request URL, query string included. [`RedactingHttpClient`] delegates
//! every request to rmcp's own implementation for `reqwest::Client` and
//! strips the URL from each client error with
//! [`reqwest::Error::without_url`], so neither those log lines nor the
//! errors handed back to this crate can contain endpoint secrets.

use std::collections::HashMap;
use std::sync::Arc;

use http::{HeaderName, HeaderValue};
use reqwest_013 as reqwest;
use rmcp::model::ClientJsonRpcMessage;
use rmcp::transport::common::client_side_sse::BoxedSseResponse;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
    StreamableHttpResponseLimits,
};

type HttpError = StreamableHttpError<reqwest::Error>;

/// Maximum number of redirects followed for one request, as in rmcp.
const MAX_REDIRECTS: usize = 10;

/// rmcp's default reqwest client wrapped so that errors carry no URL.
#[derive(Clone, Debug)]
pub(crate) struct RedactingHttpClient {
    inner: reqwest::Client,
}

impl RedactingHttpClient {
    /// Builds the client exactly as rmcp's default: no idle connection
    /// pooling and [`same_origin_redirect_policy`].
    pub(crate) fn new() -> std::result::Result<Self, reqwest::Error> {
        let inner = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .redirect(same_origin_redirect_policy())
            .build()
            .map_err(reqwest::Error::without_url)?;
        Ok(Self { inner })
    }
}

/// rmcp's default redirect policy (private there, so reimplemented): follow
/// only same-origin 307 and 308 redirects, which keep the method and body,
/// for at most [`MAX_REDIRECTS`] hops. Anything else is returned to the
/// transport as the redirect response, so credentials and session ids are
/// never replayed to another origin.
fn same_origin_redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        let preserves_body = matches!(
            attempt.status(),
            reqwest::StatusCode::TEMPORARY_REDIRECT | reqwest::StatusCode::PERMANENT_REDIRECT
        );
        let same_origin = attempt
            .previous()
            .first()
            .is_some_and(|original| original.origin() == attempt.url().origin());
        if !preserves_body || !same_origin {
            attempt.stop()
        } else if attempt.previous().len() > MAX_REDIRECTS {
            attempt.error(format!("exceeded {MAX_REDIRECTS} redirects"))
        } else {
            attempt.follow()
        }
    })
}

/// Drops the request URL from a client error; other variants carry none.
fn redact(error: HttpError) -> HttpError {
    match error {
        StreamableHttpError::Client(e) => StreamableHttpError::Client(e.without_url()),
        other => other,
    }
}

impl StreamableHttpClient for RedactingHttpClient {
    type Error = reqwest::Error;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, HttpError> {
        self.inner
            .post_message(uri, message, session_id, auth_header, custom_headers)
            .await
            .map_err(redact)
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, HttpError> {
        self.inner
            .post_message_with_max_sse_event_size(
                uri,
                message,
                session_id,
                auth_header,
                custom_headers,
                max_sse_event_size,
            )
            .await
            .map_err(redact)
    }

    async fn post_message_with_response_limits(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        limits: StreamableHttpResponseLimits,
    ) -> Result<StreamableHttpPostResponse, HttpError> {
        self.inner
            .post_message_with_response_limits(
                uri,
                message,
                session_id,
                auth_header,
                custom_headers,
                limits,
            )
            .await
            .map_err(redact)
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), HttpError> {
        self.inner
            .delete_session(uri, session_id, auth_header, custom_headers)
            .await
            .map_err(redact)
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxedSseResponse, HttpError> {
        self.inner
            .get_stream(uri, session_id, last_event_id, auth_header, custom_headers)
            .await
            .map_err(redact)
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxedSseResponse, HttpError> {
        self.inner
            .get_stream_with_max_sse_event_size(
                uri,
                session_id,
                last_event_id,
                auth_header,
                custom_headers,
                max_sse_event_size,
            )
            .await
            .map_err(redact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr").port()
    }

    fn ping() -> ClientJsonRpcMessage {
        serde_json::from_value(serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
            .expect("ping message")
    }

    #[tokio::test]
    async fn errors_carry_no_url() {
        let port = closed_port();
        let uri: Arc<str> = format!("http://127.0.0.1:{port}/mcp?token=secret123").into();
        let client = RedactingHttpClient::new().expect("client");

        let post = client
            .post_message(uri.clone(), ping(), None, None, HashMap::new())
            .await
            .expect_err("nothing is listening");
        let get = client
            .get_stream(uri.clone(), Some("s".into()), None, None, HashMap::new())
            .await
            .err()
            .expect("nothing is listening");
        let delete = client
            .delete_session(uri, "s".into(), None, HashMap::new())
            .await
            .expect_err("nothing is listening");
        for err in [post, get, delete] {
            assert!(matches!(err, StreamableHttpError::Client(_)), "{err:?}");
            for text in [err.to_string(), format!("{err:?}")] {
                assert!(!text.contains("secret123"), "query leaked: {text}");
                assert!(!text.contains("/mcp"), "url leaked: {text}");
            }
        }
    }

    /// Serves `/start` (a `status` redirect to `location`) and counts hits on
    /// `/target`; returns the base URL and the counter.
    async fn redirect_server(
        status: u16,
        location: Option<String>,
    ) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let target_hits = hits.clone();
        let location = location.unwrap_or_else(|| format!("{base}/target"));
        let app = axum::Router::new()
            .route(
                "/start",
                axum::routing::post(move || {
                    let location = location.clone();
                    async move {
                        (
                            http::StatusCode::from_u16(status).unwrap(),
                            [(http::header::LOCATION, location)],
                        )
                    }
                }),
            )
            .route(
                "/target",
                axum::routing::any(move || {
                    target_hits.fetch_add(1, Ordering::SeqCst);
                    async { http::StatusCode::ACCEPTED }
                }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, hits)
    }

    #[tokio::test]
    async fn follows_only_same_origin_body_preserving_redirects() {
        use std::sync::atomic::Ordering;
        let client = RedactingHttpClient::new().expect("client");
        let post = |base: String| {
            let client = client.clone();
            async move {
                let uri: Arc<str> = format!("{base}/start").into();
                let _ = client
                    .post_message(uri, ping(), None, None, HashMap::new())
                    .await;
            }
        };

        let (base, hits) = redirect_server(307, None).await;
        post(base).await;
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "same-origin 307 is followed"
        );

        let (base, hits) = redirect_server(302, None).await;
        post(base).await;
        assert_eq!(hits.load(Ordering::SeqCst), 0, "302 is not followed");

        let (other, other_hits) = redirect_server(307, None).await;
        let (base, _) = redirect_server(307, Some(format!("{other}/target"))).await;
        post(base).await;
        assert_eq!(
            other_hits.load(Ordering::SeqCst),
            0,
            "cross-origin 307 is not followed"
        );
    }

    #[test]
    fn redact_leaves_other_variants() {
        let err = redact(StreamableHttpError::UnexpectedEndOfStream);
        assert!(matches!(err, StreamableHttpError::UnexpectedEndOfStream));
    }
}
