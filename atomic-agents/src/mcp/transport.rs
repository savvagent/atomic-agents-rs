//! Transport selection and connection setup for MCP sessions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;

use rmcp::model::{ClientCapabilities, ClientConfig, Implementation, ProtocolVersion};
use rmcp::service::{ClientCacheConfig, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};

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

/// How to reach an MCP server.
///
/// URLs in the HTTP variants are used verbatim; use
/// [`McpTransport::from_endpoint`] for upstream-compatible `/mcp` / `/sse`
/// suffix resolution.
#[derive(Clone, PartialEq, Eq)]
pub enum McpTransport {
    /// Spawn `command` and speak MCP over its stdin/stdout. The child's
    /// stderr is discarded so it cannot clutter the host terminal.
    Stdio {
        /// Program to run.
        command: String,
        /// Arguments passed to the program.
        args: Vec<String>,
        /// Working directory for the child, if not the current one.
        cwd: Option<PathBuf>,
        /// Extra environment variables for the child.
        env: Vec<(String, String)>,
    },
    /// Streamable HTTP at `url`.
    StreamableHttp {
        /// Full endpoint URL.
        url: String,
        /// Extra request headers.
        headers: Vec<(String, String)>,
    },
    /// Legacy HTTP + SSE at `url`.
    Sse {
        /// Full SSE stream URL.
        url: String,
        /// Extra request headers.
        headers: Vec<(String, String)>,
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
        }
    }

    /// A Streamable HTTP transport for `url` (used verbatim).
    pub fn streamable_http(url: impl Into<String>) -> Self {
        Self::StreamableHttp {
            url: url.into(),
            headers: Vec::new(),
        }
    }

    /// A legacy SSE transport for `url` (used verbatim).
    pub fn sse(url: impl Into<String>) -> Self {
        Self::Sse {
            url: url.into(),
            headers: Vec::new(),
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

    fn kind_label(&self) -> String {
        match self {
            Self::Stdio { command, .. } => format!("stdio '{command}'"),
            Self::StreamableHttp { url, .. } => format!("streamable HTTP {}", redact_url(url)),
            Self::Sse { url, .. } => format!("SSE {}", redact_url(url)),
        }
    }
}

/// Formats the transport for diagnostics. URLs are redacted with
/// [`redact_url`], and header and environment values are omitted, because
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
            } => f
                .debug_struct("Stdio")
                .field("command", command)
                .field("args", args)
                .field("cwd", cwd)
                .field("env_names", &names(env))
                .finish(),
            Self::StreamableHttp { url, headers } => f
                .debug_struct("StreamableHttp")
                .field("url", &redact_url(url))
                .field("header_names", &names(headers))
                .finish(),
            Self::Sse { url, headers } => f
                .debug_struct("Sse")
                .field("url", &redact_url(url))
                .field("header_names", &names(headers))
                .finish(),
        }
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

/// Splits headers into a bearer token (if any) and the remaining header map.
fn http_headers(
    headers: &[(String, String)],
) -> Result<(Option<String>, HashMap<http::HeaderName, http::HeaderValue>)> {
    let mut bearer = None;
    let mut map = HashMap::new();
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("authorization")
            && let Some(token) = value.strip_prefix("Bearer ")
        {
            bearer = Some(token.to_string());
            continue;
        }
        let name = http::HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| Error::Mcp(format!("invalid header name {name:?}: {e}")))?;
        let value = http::HeaderValue::from_str(value)
            .map_err(|e| Error::Mcp(format!("invalid value for header {name}: {e}")))?;
        map.insert(name, value);
    }
    Ok((bearer, map))
}

/// Opens `transport` and completes the MCP initialize handshake.
pub(crate) async fn connect_transport(
    transport: &McpTransport,
) -> Result<RunningService<RoleClient, ClientConfig>> {
    let kind = transport.kind_label();
    let fail = |e: &dyn std::fmt::Display| {
        Error::Mcp(format!("failed to connect to MCP server ({kind}): {e}"))
    };
    let service = match transport {
        McpTransport::Stdio {
            command,
            args,
            cwd,
            env,
        } => {
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(args).envs(env.iter().map(|(k, v)| (k, v)));
            if let Some(dir) = cwd {
                cmd.current_dir(dir);
            }
            let (child, _stderr) = TokioChildProcess::builder(cmd)
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| fail(&e))?;
            client_config().serve(child).await.map_err(|e| fail(&e))?
        }
        McpTransport::StreamableHttp { url, headers } => {
            let (bearer, custom) = http_headers(headers).map_err(|e| fail(&e))?;
            let mut cfg =
                StreamableHttpClientTransportConfig::with_uri(url.as_str()).custom_headers(custom);
            if let Some(token) = bearer {
                cfg = cfg.auth_header(token);
            }
            let http = StreamableHttpClientTransport::from_config(cfg);
            client_config().serve(http).await.map_err(|e| fail(&e))?
        }
        McpTransport::Sse { .. } => {
            return Err(fail(&"legacy SSE transport not yet available"));
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
            }
        );
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
