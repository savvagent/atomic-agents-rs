//! Persistent MCP client sessions.

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rmcp::RoleClient;
use rmcp::model::{
    CallToolRequestParams, ClientConfig, ErrorCode, PaginatedRequestParams, ServerCapabilities,
    ServerPeerInfo,
};
use rmcp::service::{Peer, RunningService, ServiceError};
use serde_json::{Map, Value};

use super::definitions::{McpPromptDefinition, McpResourceDefinition, McpToolDefinition};
use super::result::extract_tool_result;
use super::transport::{McpTransport, McpTransportType, connect_transport};
use crate::error::{Error, Result};

/// One initialised MCP session.
///
/// Connecting performs the `initialize` handshake once; every later request
/// reuses the session. Clones share the session. It ends on
/// [`McpClient::close`], or when the last clone is dropped (a stdio child is
/// then killed).
///
/// Discovery is gated by the server's advertised capabilities: a server that
/// does not advertise `resources` or `prompts` never receives the
/// corresponding list request, and the method returns an empty list.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

struct Inner {
    peer: Peer<RoleClient>,
    info: Arc<ServerPeerInfo>,
    closed: AtomicBool,
    service: tokio::sync::Mutex<Option<RunningService<RoleClient, ClientConfig>>>,
}

impl McpClient {
    /// Connects over `transport` and completes the initialize handshake.
    pub async fn connect(transport: McpTransport) -> Result<Self> {
        let service = connect_transport(&transport).await?;
        let info = service.peer_info().ok_or_else(|| {
            Error::Mcp("MCP server did not complete the initialize handshake".to_string())
        })?;
        Ok(Self {
            inner: Arc::new(Inner {
                peer: service.peer().clone(),
                info,
                closed: AtomicBool::new(false),
                service: tokio::sync::Mutex::new(Some(service)),
            }),
        })
    }

    /// Connects to an upstream-style endpoint string; see
    /// [`McpTransport::from_endpoint`] for how it is interpreted.
    pub async fn connect_endpoint(endpoint: &str, kind: McpTransportType) -> Result<Self> {
        Self::connect(McpTransport::from_endpoint(endpoint, kind)?).await
    }

    /// Capabilities the server advertised in `initialize`.
    pub fn server_capabilities(&self) -> &ServerCapabilities {
        &self.inner.info.capabilities
    }

    /// The negotiated protocol version, e.g. `2025-11-25`.
    pub fn protocol_version(&self) -> String {
        self.inner.info.protocol_version.to_string()
    }

    /// The server's implementation name, when it sent one.
    pub fn server_name(&self) -> Option<String> {
        self.inner.info.server_info.as_ref().map(|i| i.name.clone())
    }

    fn peer(&self) -> Result<&Peer<RoleClient>> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(Error::Mcp("MCP session is closed".to_string()));
        }
        Ok(&self.inner.peer)
    }

    /// Lists every tool, following pagination. Empty when the server does not
    /// advertise the `tools` capability.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>> {
        let peer = self.peer()?;
        if self.server_capabilities().tools.is_none() {
            return Ok(Vec::new());
        }
        let tools = collect_pages("tools/list", |cursor| async move {
            let page = peer.list_tools(page_params(cursor)).await?;
            Ok((page.tools, page.next_cursor))
        })
        .await?;
        Ok(tools.into_iter().map(McpToolDefinition::from).collect())
    }

    /// Lists every resource followed by every resource template. Empty when
    /// the server does not advertise the `resources` capability. A server that
    /// does not implement `resources/templates/list` (`METHOD_NOT_FOUND`) is
    /// treated as having no templates.
    pub async fn list_resources(&self) -> Result<Vec<McpResourceDefinition>> {
        let peer = self.peer()?;
        if self.server_capabilities().resources.is_none() {
            return Ok(Vec::new());
        }
        let resources = collect_pages("resources/list", |cursor| async move {
            let page = peer.list_resources(page_params(cursor)).await?;
            Ok((page.resources, page.next_cursor))
        })
        .await?;
        let templates = collect_pages("resources/templates/list", |cursor| async move {
            let first = cursor.is_none();
            match peer.list_resource_templates(page_params(cursor)).await {
                Ok(page) => Ok((page.resource_templates, page.next_cursor)),
                Err(ServiceError::McpError(e))
                    if first && e.code == ErrorCode::METHOD_NOT_FOUND =>
                {
                    Ok((Vec::new(), None))
                }
                Err(e) => Err(e),
            }
        })
        .await?;
        Ok(resources
            .into_iter()
            .map(McpResourceDefinition::from_resource)
            .chain(
                templates
                    .into_iter()
                    .map(McpResourceDefinition::from_template),
            )
            .collect())
    }

    /// Lists every prompt, following pagination. Empty when the server does
    /// not advertise the `prompts` capability.
    pub async fn list_prompts(&self) -> Result<Vec<McpPromptDefinition>> {
        let peer = self.peer()?;
        if self.server_capabilities().prompts.is_none() {
            return Ok(Vec::new());
        }
        let prompts = collect_pages("prompts/list", |cursor| async move {
            let page = peer.list_prompts(page_params(cursor)).await?;
            Ok((page.prompts, page.next_cursor))
        })
        .await?;
        Ok(prompts.into_iter().map(McpPromptDefinition::from).collect())
    }

    /// Calls tool `name` with `args` and converts the result to JSON.
    ///
    /// A non-null `structuredContent` is returned as-is; a single text block
    /// is parsed as JSON when possible and otherwise returned as a string;
    /// any other content is returned as the MCP content array. A result with
    /// `isError: true` and JSON-RPC errors (such as an unknown tool) are `Err`.
    pub async fn call_tool(&self, name: &str, args: Map<String, Value>) -> Result<Value> {
        let peer = self.peer()?;
        let result = peer
            .call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(args))
            .await
            .map_err(|e| Error::Mcp(format!("tools/call '{name}' failed: {e}")))?;
        extract_tool_result(name, result)
    }

    /// Gracefully ends the session for every clone. Idempotent; afterwards
    /// every request returns an error.
    pub async fn close(&self) -> Result<()> {
        self.inner.closed.store(true, Ordering::SeqCst);
        let service = self.inner.service.lock().await.take();
        if let Some(mut service) = service {
            service
                .close()
                .await
                .map_err(|e| Error::Mcp(format!("failed to close MCP session: {e}")))?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("protocol_version", &self.protocol_version())
            .field("server_name", &self.server_name())
            .field("closed", &self.inner.closed.load(Ordering::SeqCst))
            .finish()
    }
}

fn page_params(cursor: Option<String>) -> Option<PaginatedRequestParams> {
    cursor.map(|c| PaginatedRequestParams::default().with_cursor(Some(c)))
}

/// Errors raised while paginating an MCP list response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpPaginationError {
    /// The server returned a `nextCursor` it had already returned.
    RepeatedCursor(String),
}

impl std::fmt::Display for McpPaginationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepeatedCursor(c) => {
                write!(f, "MCP pagination returned a repeated cursor: {c:?}")
            }
        }
    }
}

impl std::error::Error for McpPaginationError {}

impl From<McpPaginationError> for Error {
    fn from(e: McpPaginationError) -> Self {
        Error::Mcp(e.to_string())
    }
}

/// Follows opaque `nextCursor` values, concatenating every page's items.
///
/// rmcp's `list_all_*` helpers do not detect cycles, so this loop stops with
/// [`McpPaginationError::RepeatedCursor`] when a cursor repeats, so a
/// misbehaving server cannot cause an infinite loop. Errors from `fetch`
/// (including continuation-page failures) are returned with `method` as context.
pub(crate) async fn collect_pages<T, F, Fut>(method: &str, mut fetch: F) -> Result<Vec<T>>
where
    F: FnMut(Option<String>) -> Fut,
    Fut: Future<Output = std::result::Result<(Vec<T>, Option<String>), ServiceError>>,
{
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let mut cursor = None;
    loop {
        let (page, next) = fetch(cursor.take())
            .await
            .map_err(|e| Error::Mcp(format!("{method} failed: {e}")))?;
        items.extend(page);
        match next {
            Some(next) => {
                if !seen.insert(next.clone()) {
                    return Err(McpPaginationError::RepeatedCursor(next).into());
                }
                cursor = Some(next);
            }
            None => return Ok(items),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    type Page = std::result::Result<(Vec<u32>, Option<String>), ServiceError>;
    type CursorLog = Arc<Mutex<Vec<Option<String>>>>;

    /// A fetcher that serves `pages` in order and records the cursors it got.
    fn scripted(
        pages: Vec<Page>,
    ) -> (
        CursorLog,
        impl FnMut(Option<String>) -> std::future::Ready<Page>,
    ) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let mut pages = pages.into_iter();
        let fetch = move |cursor| {
            log.lock().unwrap().push(cursor);
            std::future::ready(pages.next().expect("fetched past the last page"))
        };
        (seen, fetch)
    }

    #[tokio::test]
    async fn follows_cursors_and_merges_pages() {
        let (cursors, fetch) = scripted(vec![
            Ok((vec![1, 2], Some("c1".into()))),
            Ok((vec![3], Some("c2".into()))),
            Ok((vec![4], None)),
        ]);
        assert_eq!(collect_pages("x/list", fetch).await.unwrap(), [1, 2, 3, 4]);
        assert_eq!(
            *cursors.lock().unwrap(),
            [None, Some("c1".to_string()), Some("c2".to_string())]
        );
    }

    #[tokio::test]
    async fn repeated_cursor_is_an_error() {
        let (_, fetch) = scripted(vec![
            Ok((vec![1], Some("c1".into()))),
            Ok((vec![2], Some("c1".into()))),
        ]);
        let err = collect_pages("x/list", fetch).await.unwrap_err();
        let expected: Error = McpPaginationError::RepeatedCursor("c1".into()).into();
        assert_eq!(err.to_string(), expected.to_string());
        assert!(err.to_string().contains("repeated cursor"), "{err}");
    }

    #[tokio::test]
    async fn continuation_error_propagates() {
        let (_, fetch) = scripted(vec![
            Ok((vec![1], Some("c1".into()))),
            Err(ServiceError::TransportClosed),
        ]);
        let err = collect_pages("x/list", fetch).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("x/list failed") && msg.contains("Transport closed"),
            "{msg}"
        );
    }

    #[test]
    fn first_page_sends_no_params() {
        assert!(page_params(None).is_none());
        assert_eq!(
            page_params(Some("c".into()))
                .and_then(|p| p.cursor)
                .as_deref(),
            Some("c")
        );
    }
}
