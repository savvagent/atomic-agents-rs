//! Transport selection and connection setup for MCP sessions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use rmcp::model::{ClientCapabilities, ClientConfig, Implementation, ProtocolVersion};
use rmcp::service::{ClientCacheConfig, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};

use super::legacy_sse::{LegacySseTransport, Limits};
use super::redacting_http::RedactingHttpClient;
use crate::error::{Error, Result};

/// MCP transport protocol, used with endpoint strings
/// ([`McpTransport::from_endpoint`], [`McpClient::connect_endpoint`](super::McpClient::connect_endpoint)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpTransportType {
    /// HTTP + Server-Sent Events (legacy, protocol 2024-11-05).
    Sse,
    /// Streamable HTTP (recommended).
    HttpStream,
    /// Standard I/O (command-line process).
    Stdio,
}

/// Parent environment variables a stdio child receives by default, as in the
/// upstream Python MCP SDK's `get_default_environment()`. A value starting
/// with `()` (an exported shell function) is never passed.
#[cfg(not(windows))]
pub const DEFAULT_INHERITED_ENV_VARS: &[&str] =
    &["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"];

/// Parent environment variables a stdio child receives by default, as in the
/// upstream Python MCP SDK's `get_default_environment()`. A value starting
/// with `()` (an exported shell function) is never passed.
#[cfg(windows)]
pub const DEFAULT_INHERITED_ENV_VARS: &[&str] = &[
    "APPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "PATH",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "USERNAME",
    "USERPROFILE",
];

/// Default time allowed for each MCP request; see
/// [`McpTransport::with_request_timeout`].
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Default time allowed to connect and complete `initialize`; see
/// [`McpTransport::with_connect_timeout`].
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How to reach an MCP server.
///
/// URLs in the HTTP variants are used verbatim; use
/// [`McpTransport::from_endpoint`] for upstream-compatible `/mcp` / `/sse`
/// suffix resolution.
///
/// Every variant carries a request timeout and a connect timeout
/// ([`DEFAULT_REQUEST_TIMEOUT`], [`DEFAULT_CONNECT_TIMEOUT`] unless set).
#[derive(Clone, PartialEq, Eq)]
pub enum McpTransport {
    /// Spawn `command` and speak MCP over its stdin/stdout. The child's
    /// stderr is discarded so it cannot clutter the host terminal.
    ///
    /// The child does not inherit the parent's environment (which may hold
    /// LLM API keys and other secrets): it gets only the variables in
    /// [`DEFAULT_INHERITED_ENV_VARS`] that are set in the parent, then `env`.
    /// Set `inherit_env` (see [`McpTransport::with_inherited_env`]) to pass
    /// the full parent environment instead.
    Stdio {
        /// Program to run.
        command: String,
        /// Arguments passed to the program.
        args: Vec<String>,
        /// Working directory for the child, if not the current one.
        cwd: Option<PathBuf>,
        /// Extra environment variables for the child.
        env: Vec<(String, String)>,
        /// Pass the parent's full environment rather than the allowlist.
        inherit_env: bool,
        /// Time allowed for each request.
        request_timeout: Duration,
        /// Time allowed to spawn the child and complete `initialize`.
        connect_timeout: Duration,
    },
    /// Streamable HTTP at `url`.
    StreamableHttp {
        /// Full endpoint URL.
        url: String,
        /// Extra request headers.
        headers: Vec<(String, String)>,
        /// Time allowed for each request.
        request_timeout: Duration,
        /// Time allowed to complete `initialize`.
        connect_timeout: Duration,
    },
    /// Legacy HTTP + SSE at `url`.
    Sse {
        /// Full SSE stream URL.
        url: String,
        /// Extra request headers.
        headers: Vec<(String, String)>,
        /// Time allowed for each request.
        request_timeout: Duration,
        /// Time allowed to open the stream, receive the `endpoint` event and
        /// complete `initialize`.
        connect_timeout: Duration,
    },
}

impl McpTransport {
    /// A stdio transport running `command` with `args`.
    pub fn stdio(command: impl Into<String>, args: Vec<String>) -> Self {
        Self::Stdio {
            command: command.into(),
            args,
            cwd: None,
            env: Vec::new(),
            inherit_env: false,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    /// A Streamable HTTP transport for `url` (used verbatim).
    pub fn streamable_http(url: impl Into<String>) -> Self {
        Self::StreamableHttp {
            url: url.into(),
            headers: Vec::new(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    /// A legacy SSE transport for `url` (used verbatim).
    pub fn sse(url: impl Into<String>) -> Self {
        Self::Sse {
            url: url.into(),
            headers: Vec::new(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    /// Sets the time allowed for each request (each page of a list, a tool
    /// call, a resource read, a prompt render). On expiry the request is
    /// abandoned (rmcp sends `notifications/cancelled`) and an error is
    /// returned; the session stays usable. Cancellation is advisory: a server
    /// may already have applied a tool call's side effects, so treat a
    /// timed-out call as possibly applied before retrying. Applies to every
    /// variant.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        match &mut self {
            Self::Stdio {
                request_timeout, ..
            }
            | Self::StreamableHttp {
                request_timeout, ..
            }
            | Self::Sse {
                request_timeout, ..
            } => *request_timeout = timeout,
        }
        self
    }

    /// Sets the time allowed to connect: spawning the child or opening the
    /// stream, plus the `initialize` handshake. Applies to every variant.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        match &mut self {
            Self::Stdio {
                connect_timeout, ..
            }
            | Self::StreamableHttp {
                connect_timeout, ..
            }
            | Self::Sse {
                connect_timeout, ..
            } => *connect_timeout = timeout,
        }
        self
    }

    /// The time allowed for each request.
    pub fn request_timeout(&self) -> Duration {
        match self {
            Self::Stdio {
                request_timeout, ..
            }
            | Self::StreamableHttp {
                request_timeout, ..
            }
            | Self::Sse {
                request_timeout, ..
            } => *request_timeout,
        }
    }

    /// The time allowed to connect and complete `initialize`.
    pub fn connect_timeout(&self) -> Duration {
        match self {
            Self::Stdio {
                connect_timeout, ..
            }
            | Self::StreamableHttp {
                connect_timeout, ..
            }
            | Self::Sse {
                connect_timeout, ..
            } => *connect_timeout,
        }
    }

    /// Sets the child's working directory.
    ///
    /// Only meaningful for [`McpTransport::Stdio`]; HTTP transports are
    /// returned unchanged.
    pub fn with_cwd(mut self, dir: impl Into<PathBuf>) -> Self {
        if let Self::Stdio { cwd, .. } = &mut self {
            *cwd = Some(dir.into());
        }
        self
    }

    /// Adds an environment variable for the child.
    ///
    /// Only meaningful for [`McpTransport::Stdio`]; HTTP transports are
    /// returned unchanged.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let Self::Stdio { env, .. } = &mut self {
            env.push((key.into(), value.into()));
        }
        self
    }

    /// Passes the parent's full environment to the child instead of only
    /// [`DEFAULT_INHERITED_ENV_VARS`]. Variables from
    /// [`with_env`](Self::with_env) still apply on top.
    ///
    /// Only meaningful for [`McpTransport::Stdio`]; HTTP transports are
    /// returned unchanged.
    pub fn with_inherited_env(mut self) -> Self {
        if let Self::Stdio { inherit_env, .. } = &mut self {
            *inherit_env = true;
        }
        self
    }

    /// Adds a request header. `Authorization: Bearer <token>` is sent through
    /// the transport's bearer-auth support.
    ///
    /// Only meaningful for the HTTP transports; [`McpTransport::Stdio`] is
    /// returned unchanged.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        if let Self::StreamableHttp { headers, .. } | Self::Sse { headers, .. } = &mut self {
            headers.push((name.into(), value.into()));
        }
        self
    }

    /// Builds a transport from an upstream-style endpoint string.
    ///
    /// - `Stdio`: the endpoint is a shell-quoted command line, split with
    ///   POSIX shell rules (`python -m 'my server'` → `python`, `-m`, `my server`).
    /// - `HttpStream`: `/mcp` is appended to the URL path unless it already ends with it.
    /// - `Sse`: `/sse` is appended to the URL path unless it already ends with it.
    ///
    /// Query strings and fragments are preserved.
    pub fn from_endpoint(endpoint: &str, kind: McpTransportType) -> Result<Self> {
        match kind {
            McpTransportType::Stdio => {
                let mut parts = shlex::split(endpoint)
                    .filter(|p| !p.is_empty())
                    .ok_or_else(|| {
                        Error::Mcp(
                            "STDIO endpoint command is empty or has unbalanced quotes".to_string(),
                        )
                    })?;
                let command = parts.remove(0);
                Ok(Self::stdio(command, parts))
            }
            McpTransportType::HttpStream => {
                Ok(Self::streamable_http(resolve_endpoint(endpoint, "/mcp")?))
            }
            McpTransportType::Sse => Ok(Self::sse(resolve_endpoint(endpoint, "/sse")?)),
        }
    }

    /// The scrubber for error text produced while talking over this transport.
    pub(crate) fn scrubber(&self) -> UrlScrubber {
        match self {
            Self::Stdio { .. } => UrlScrubber::none(),
            Self::StreamableHttp { url, .. } | Self::Sse { url, .. } => UrlScrubber::new(url),
        }
    }

    fn kind_label(&self) -> String {
        match self {
            Self::Stdio { command, .. } => format!("stdio '{command}'"),
            Self::StreamableHttp { url, .. } => format!("streamable HTTP {}", redact_url(url)),
            Self::Sse { url, .. } => format!("SSE {}", redact_url(url)),
        }
    }
}

/// Formats the transport for diagnostics. URLs are redacted with
/// `redact_url`, and header and environment values are omitted, because
/// all of them commonly carry credentials.
impl std::fmt::Debug for McpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = |pairs: &[(String, String)]| -> Vec<String> {
            pairs.iter().map(|(k, _)| k.clone()).collect()
        };
        match self {
            Self::Stdio {
                command,
                args,
                cwd,
                env,
                inherit_env,
                request_timeout,
                connect_timeout,
            } => f
                .debug_struct("Stdio")
                .field("command", command)
                .field("args", &format_args!("<{}>", args.len()))
                .field("cwd", cwd)
                .field("env_names", &names(env))
                .field("inherit_env", inherit_env)
                .field("request_timeout", request_timeout)
                .field("connect_timeout", connect_timeout)
                .finish(),
            Self::StreamableHttp {
                url,
                headers,
                request_timeout,
                connect_timeout,
            } => f
                .debug_struct("StreamableHttp")
                .field("url", &redact_url(url))
                .field("header_names", &names(headers))
                .field("request_timeout", request_timeout)
                .field("connect_timeout", connect_timeout)
                .finish(),
            Self::Sse {
                url,
                headers,
                request_timeout,
                connect_timeout,
            } => f
                .debug_struct("Sse")
                .field("url", &redact_url(url))
                .field("header_names", &names(headers))
                .field("request_timeout", request_timeout)
                .field("connect_timeout", connect_timeout)
                .finish(),
        }
    }
}

/// Removes URL secrets from error strings produced by lower layers.
///
/// HTTP clients (reqwest) append the full request URL to their errors, so the
/// query string, fragment and userinfo must be scrubbed from the formatted
/// text, not only from our own labels. A scrubber built from a non-URL
/// (stdio) is a no-op.
#[derive(Clone, Debug, Default)]
pub(crate) struct UrlScrubber {
    /// `(needle, replacement)` pairs, applied in order (longest first).
    rules: Vec<(String, String)>,
}

impl UrlScrubber {
    /// A scrubber that changes nothing.
    pub(crate) fn none() -> Self {
        Self::default()
    }

    /// Builds a scrubber for `url`. An unparseable URL yields a no-op.
    pub(crate) fn new(url: &str) -> Self {
        let Ok(parsed) = url::Url::parse(url) else {
            return Self::none();
        };
        let redacted = redact_url(url);
        let mut rules = Vec::new();
        let mut push = |needle: String, with: String| {
            if !needle.is_empty() && !rules.iter().any(|(n, _)| *n == needle) {
                rules.push((needle, with));
            }
        };
        let mut no_userinfo = parsed.clone();
        let _ = no_userinfo.set_username("");
        let _ = no_userinfo.set_password(None);
        push(url.to_string(), redacted.clone());
        push(parsed.to_string(), redacted.clone());
        push(no_userinfo.to_string(), redacted);
        if !parsed.username().is_empty() || parsed.password().is_some() {
            let info = match parsed.password() {
                Some(pw) => format!("{}:{pw}@", parsed.username()),
                None => format!("{}@", parsed.username()),
            };
            push(info, String::new());
        }
        // An empty `?`/`#` would make every `?`/`#` in the text a match.
        if let Some(q) = parsed.query().filter(|q| !q.is_empty()) {
            push(format!("?{q}"), "?<redacted>".to_string());
        }
        if let Some(f) = parsed.fragment().filter(|f| !f.is_empty()) {
            push(format!("#{f}"), "#<redacted>".to_string());
        }
        rules.sort_by_key(|(n, _)| std::cmp::Reverse(n.len()));
        Self { rules }
    }

    /// Returns `text` with every known secret-bearing part replaced.
    pub(crate) fn scrub(&self, text: &str) -> String {
        self.rules
            .iter()
            .fold(text.to_string(), |acc, (n, r)| acc.replace(n.as_str(), r))
    }
}

/// Strips everything from `url` that may carry credentials (userinfo, query
/// string, fragment) so it can appear in errors and logs. Falls back to
/// `"<invalid url>"` when it does not parse.
pub(crate) fn redact_url(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut u) => {
            let _ = u.set_username("");
            let _ = u.set_password(None);
            u.set_query(None);
            u.set_fragment(None);
            u.to_string()
        }
        Err(_) => "<invalid url>".to_string(),
    }
}

/// Appends `suffix` to `endpoint`'s path unless the path already ends with it.
///
/// Mirrors upstream `_transport_endpoint`: trailing slashes are ignored when
/// comparing, and the query string and fragment are kept.
pub(crate) fn resolve_endpoint(endpoint: &str, suffix: &str) -> Result<String> {
    let mut url =
        url::Url::parse(endpoint).map_err(|e| Error::Mcp(format!("Invalid endpoint URL: {e}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::Mcp(format!(
            "Invalid endpoint URL: unsupported scheme '{}'",
            url.scheme()
        )));
    }
    let path = url.path().trim_end_matches('/').to_string();
    if path.ends_with(suffix) {
        return Ok(endpoint.to_string());
    }
    url.set_path(&format!("{path}{suffix}"));
    Ok(url.to_string())
}

/// Client identity and protocol settings sent in `initialize`.
fn client_config() -> ClientConfig {
    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("atomic-agents", env!("CARGO_PKG_VERSION")),
    )
    // Pin the newest version that still uses the initialize handshake, so
    // every session performs exactly one `initialize`.
    .with_protocol_version(ProtocolVersion::LATEST_WITH_INITIALIZE)
}

/// Parses `headers` into validated pairs, in order, with values marked
/// sensitive so they are redacted from `Debug` output. Shared by both HTTP
/// transports.
fn parse_header_pairs(
    headers: &[(String, String)],
) -> Result<Vec<(http::HeaderName, http::HeaderValue)>> {
    headers
        .iter()
        .map(|(name, value)| {
            let name = http::HeaderName::from_bytes(name.as_bytes())
                .map_err(|e| Error::Mcp(format!("invalid header name {name:?}: {e}")))?;
            let mut value = http::HeaderValue::from_str(value)
                .map_err(|e| Error::Mcp(format!("invalid value for header {name}: {e}")))?;
            value.set_sensitive(true);
            Ok((name, value))
        })
        .collect()
}

/// Builds a header map for the legacy SSE transport. Repeated names are all
/// kept (`append`), and `Authorization` is sent as given.
fn sse_header_map(headers: &[(String, String)]) -> Result<http::HeaderMap> {
    let mut map = http::HeaderMap::new();
    for (name, value) in parse_header_pairs(headers)? {
        map.append(name, value);
    }
    Ok(map)
}

/// Splits headers into a bearer token (if any) and the remaining header map.
/// rmcp's config takes a name-keyed map, so for the streamable HTTP transport
/// a repeated name keeps only its last value.
fn http_headers(
    headers: &[(String, String)],
) -> Result<(Option<String>, HashMap<http::HeaderName, http::HeaderValue>)> {
    let mut bearer = None;
    let mut rest = Vec::new();
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("authorization")
            && let Some(token) = value.strip_prefix("Bearer ")
        {
            bearer = Some(token.to_string());
        } else {
            rest.push((name.clone(), value.clone()));
        }
    }
    Ok((bearer, parse_header_pairs(&rest)?.into_iter().collect()))
}

/// The allowlisted parent variables for a stdio child; see
/// [`DEFAULT_INHERITED_ENV_VARS`].
fn default_child_env() -> Vec<(&'static str, std::ffi::OsString)> {
    DEFAULT_INHERITED_ENV_VARS
        .iter()
        .filter_map(|&name| Some((name, std::env::var_os(name)?)))
        .filter(|(_, value)| !value.as_encoded_bytes().starts_with(b"()"))
        .collect()
}

/// Formats `d` as seconds for error text, e.g. `60s` or `0.2s`.
pub(crate) fn secs(d: Duration) -> String {
    format!("{}s", d.as_secs_f64())
}

/// Opens `transport` and completes the MCP initialize handshake, within the
/// transport's connect timeout. On expiry the partly opened transport is
/// dropped (a stdio child is killed).
pub(crate) async fn connect_transport(
    transport: &McpTransport,
) -> Result<RunningService<RoleClient, ClientConfig>> {
    let timeout = transport.connect_timeout();
    match tokio::time::timeout(timeout, open_transport(transport)).await {
        Ok(result) => result,
        Err(_) => Err(Error::Mcp(transport.scrubber().scrub(&format!(
            "failed to connect to MCP server ({}): initialize timed out after {}",
            transport.kind_label(),
            secs(timeout)
        )))),
    }
}

async fn open_transport(
    transport: &McpTransport,
) -> Result<RunningService<RoleClient, ClientConfig>> {
    let kind = transport.kind_label();
    let scrubber = transport.scrubber();
    let fail = |e: &dyn std::fmt::Display| {
        Error::Mcp(scrubber.scrub(&format!("failed to connect to MCP server ({kind}): {e}")))
    };
    let service = match transport {
        McpTransport::Stdio {
            command,
            args,
            cwd,
            env,
            inherit_env,
            ..
        } => {
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(args);
            if !inherit_env {
                cmd.env_clear().envs(default_child_env());
            }
            cmd.envs(env.iter().map(|(k, v)| (k, v)));
            if let Some(dir) = cwd {
                cmd.current_dir(dir);
            }
            let (child, _stderr) = TokioChildProcess::builder(cmd)
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| fail(&e))?;
            client_config().serve(child).await.map_err(|e| fail(&e))?
        }
        McpTransport::StreamableHttp { url, headers, .. } => {
            let (bearer, custom) = http_headers(headers).map_err(|e| fail(&e))?;
            let mut cfg =
                StreamableHttpClientTransportConfig::with_uri(url.as_str()).custom_headers(custom);
            if let Some(token) = bearer {
                cfg = cfg.auth_header(token);
            }
            // rmcp's default client, wrapped so that no error (and so none of
            // rmcp's own log lines) carries the endpoint URL.
            let client = RedactingHttpClient::new().map_err(|e| fail(&e))?;
            let http = StreamableHttpClientTransport::with_client(client, cfg);
            client_config().serve(http).await.map_err(|e| fail(&e))?
        }
        McpTransport::Sse {
            url,
            headers,
            connect_timeout,
            request_timeout,
            ..
        } => {
            let map = sse_header_map(headers).map_err(|e| fail(&e))?;
            // Never follow redirects: they could replay credentials elsewhere.
            // A bounded connect phase only: no total timeout, since the SSE
            // stream is long-lived. The whole connect sequence is bounded by
            // the transport's connect timeout.
            let http = reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|e| fail(&e.without_url()))?;
            let limits = Limits {
                connect_timeout: *connect_timeout,
                post_timeout: *request_timeout,
                ..Limits::default()
            };
            let t = LegacySseTransport::connect_with(http, url, map, limits)
                .await
                .map_err(|e| fail(&e))?;
            // The legacy transport only exists for 2024-11-05 servers.
            client_config()
                .with_protocol_version(ProtocolVersion::V_2024_11_05)
                .serve(t)
                .await
                .map_err(|e| fail(&e))?
        }
    };
    // Discovery must always reflect the server (and repeated-cursor detection
    // must see real pages), so never answer list requests from a cache.
    service
        .peer()
        .set_response_cache_config(ClientCacheConfig::disabled())
        .await;
    Ok(service)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_appends_mcp_suffix() {
        assert_eq!(
            resolve_endpoint("http://h:1", "/mcp").unwrap(),
            "http://h:1/mcp"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/", "/mcp").unwrap(),
            "http://h:1/mcp"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/mcp", "/mcp").unwrap(),
            "http://h:1/mcp"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/mcp/", "/mcp").unwrap(),
            "http://h:1/mcp/"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/api?token=a#f", "/mcp").unwrap(),
            "http://h:1/api/mcp?token=a#f"
        );
    }

    #[test]
    fn resolve_appends_sse_suffix() {
        assert_eq!(
            resolve_endpoint("http://h:1", "/sse").unwrap(),
            "http://h:1/sse"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/", "/sse").unwrap(),
            "http://h:1/sse"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/sse", "/sse").unwrap(),
            "http://h:1/sse"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/sse/", "/sse").unwrap(),
            "http://h:1/sse/"
        );
        assert_eq!(
            resolve_endpoint("http://h:1/api?token=a#f", "/sse").unwrap(),
            "http://h:1/api/sse?token=a#f"
        );
    }

    #[test]
    fn resolve_rejects_invalid_url() {
        let err = resolve_endpoint("not a url", "/mcp").unwrap_err();
        assert!(err.to_string().contains("Invalid endpoint URL"), "{err}");
    }

    #[test]
    fn scrubber_ignores_empty_query_and_fragment() {
        let s = UrlScrubber::new("http://h:1/mcp?#");
        assert_eq!(s.scrub("why? see #3"), "why? see #3");
    }

    #[test]
    fn scrubber_removes_url_secrets_from_text() {
        let s = UrlScrubber::new("http://user:pw@127.0.0.1:9/mcp?token=secret123#frag9");
        let text = "error sending request for url (http://user:pw@127.0.0.1:9/mcp?token=secret123#frag9) \
                    and (http://127.0.0.1:9/mcp?token=secret123) again ?token=secret123 #frag9 user:pw@x";
        let out = s.scrub(text);
        for leaked in ["secret123", "pw", "user", "frag9"] {
            assert!(!out.contains(leaked), "{leaked} leaked: {out}");
        }
        assert!(out.contains("127.0.0.1:9/mcp"), "{out}");
        assert_eq!(UrlScrubber::none().scrub("a?b=c"), "a?b=c");
        assert_eq!(
            McpTransport::stdio("x", vec![]).scrubber().scrub("?q"),
            "?q"
        );
    }

    #[test]
    fn debug_hides_stdio_args() {
        let t = McpTransport::stdio("srv", vec!["--key=hunter2".into()]);
        let d = format!("{t:?}");
        assert!(!d.contains("hunter2") && d.contains("<1>"), "{d}");
    }

    #[test]
    fn resolve_rejects_non_http_schemes() {
        // `localhost:8080` parses as scheme `localhost`.
        for bad in ["localhost:8080", "ftp://h/x"] {
            let err = resolve_endpoint(bad, "/mcp").unwrap_err().to_string();
            assert!(err.contains("unsupported scheme"), "{bad}: {err}");
        }
        assert!(resolve_endpoint("https://h/x", "/mcp").is_ok());
    }

    #[test]
    fn labels_and_debug_redact_credentials() {
        let raw = "http://user:pw@h:1/mcp?token=secret#f";
        let t = McpTransport::streamable_http(raw).with_header("Authorization", "Bearer hdr");
        for text in [t.kind_label(), format!("{t:?}")] {
            assert!(text.contains("h:1/mcp"), "{text}");
            for leaked in ["secret", "pw", "user", "hdr"] {
                assert!(!text.contains(leaked), "{leaked} leaked in {text}");
            }
        }
        assert!(McpTransport::sse(raw).kind_label().contains("h:1/mcp"));
        assert_eq!(redact_url("not a url"), "<invalid url>");
    }

    #[test]
    fn from_endpoint_resolves_http_kinds() {
        assert_eq!(
            McpTransport::from_endpoint("http://h:1", McpTransportType::HttpStream).unwrap(),
            McpTransport::streamable_http("http://h:1/mcp")
        );
        assert_eq!(
            McpTransport::from_endpoint("http://h:1", McpTransportType::Sse).unwrap(),
            McpTransport::sse("http://h:1/sse")
        );
    }

    #[test]
    fn from_endpoint_stdio_uses_shell_quoting() {
        let t =
            McpTransport::from_endpoint("python -m 'my server'", McpTransportType::Stdio).unwrap();
        assert_eq!(
            t,
            McpTransport::stdio("python", vec!["-m".into(), "my server".into()])
        );
    }

    #[test]
    fn from_endpoint_stdio_rejects_empty_and_unbalanced() {
        for bad in ["", "   ", "a 'b"] {
            let err = McpTransport::from_endpoint(bad, McpTransportType::Stdio).unwrap_err();
            assert!(
                err.to_string().contains("empty or has unbalanced quotes"),
                "{bad:?}: {err}"
            );
        }
    }

    #[test]
    fn builders_only_touch_matching_variants() {
        let stdio = McpTransport::stdio("srv", vec![])
            .with_cwd("/tmp")
            .with_env("K", "V")
            .with_header("X", "y");
        assert_eq!(
            stdio,
            McpTransport::Stdio {
                command: "srv".into(),
                args: vec![],
                cwd: Some(PathBuf::from("/tmp")),
                env: vec![("K".into(), "V".into())],
                inherit_env: false,
                request_timeout: DEFAULT_REQUEST_TIMEOUT,
                connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            }
        );
        let http = McpTransport::streamable_http("http://h")
            .with_cwd("/tmp")
            .with_env("K", "V")
            .with_header("X", "y");
        assert_eq!(
            http,
            McpTransport::StreamableHttp {
                url: "http://h".into(),
                headers: vec![("X".into(), "y".into())],
                request_timeout: DEFAULT_REQUEST_TIMEOUT,
                connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            }
        );
    }

    #[test]
    fn inherited_env_only_touches_stdio() {
        let t = McpTransport::stdio("srv", vec![]).with_inherited_env();
        assert!(matches!(
            t,
            McpTransport::Stdio {
                inherit_env: true,
                ..
            }
        ));
        assert!(format!("{t:?}").contains("inherit_env: true"));
        let http = McpTransport::streamable_http("http://h");
        assert_eq!(http.clone().with_inherited_env(), http);
    }

    #[test]
    fn default_child_env_is_the_allowlist() {
        for (name, _) in default_child_env() {
            assert!(DEFAULT_INHERITED_ENV_VARS.contains(&name), "{name}");
        }
        #[cfg(not(windows))]
        assert!(default_child_env().iter().any(|(n, _)| *n == "PATH"));
    }

    #[test]
    fn timeouts_default_and_apply_to_every_variant() {
        let short = Duration::from_millis(250);
        for t in [
            McpTransport::stdio("srv", vec![]),
            McpTransport::streamable_http("http://h"),
            McpTransport::sse("http://h"),
        ] {
            assert_eq!(t.request_timeout(), DEFAULT_REQUEST_TIMEOUT);
            assert_eq!(t.connect_timeout(), DEFAULT_CONNECT_TIMEOUT);
            let t = t
                .with_request_timeout(short)
                .with_connect_timeout(2 * short);
            assert_eq!(t.request_timeout(), short);
            assert_eq!(t.connect_timeout(), 2 * short);
        }
        assert_eq!(DEFAULT_REQUEST_TIMEOUT, Duration::from_secs(60));
        assert_eq!(DEFAULT_CONNECT_TIMEOUT, Duration::from_secs(30));
        assert_eq!(secs(Duration::from_millis(200)), "0.2s");
        assert_eq!(secs(Duration::from_secs(60)), "60s");
    }

    #[test]
    fn sse_headers_append_and_reject_invalid() {
        let map = sse_header_map(&[
            ("Authorization".into(), "Bearer tok".into()),
            ("x-multi".into(), "a".into()),
            ("x-multi".into(), "b".into()),
        ])
        .unwrap();
        assert_eq!(map.get_all("x-multi").iter().count(), 2);
        assert_eq!(map["authorization"], "Bearer tok");
        assert!(sse_header_map(&[("x".into(), "a\r\nb".into())]).is_err());
    }

    #[test]
    fn headers_split_bearer_and_reject_invalid() {
        let (bearer, map) = http_headers(&[
            ("Authorization".into(), "Bearer tok".into()),
            ("x-api-key".into(), "k".into()),
        ])
        .unwrap();
        assert_eq!(bearer.as_deref(), Some("tok"));
        assert_eq!(map.len(), 1);
        assert_eq!(map[&http::HeaderName::from_static("x-api-key")], "k");
        assert!(http_headers(&[("x".into(), "a\r\nb".into())]).is_err());
        assert!(http_headers(&[("bad name".into(), "v".into())]).is_err());
    }
}
