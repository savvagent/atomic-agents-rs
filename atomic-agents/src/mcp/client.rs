//! Persistent MCP client sessions.

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rmcp::RoleClient;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ClientConfig, ErrorCode, GetPromptRequestParams,
    GetPromptResponse, PaginatedRequestParams, ReadResourceRequestParams, ReadResourceResponse,
    ServerCapabilities, ServerPeerInfo,
};
use rmcp::service::{Peer, RunningService, ServiceError};
use serde_json::{Map, Value};

use super::definitions::{McpPromptDefinition, McpResourceDefinition, McpToolDefinition};
use super::prompt::{McpPrompt, McpPromptOutput};
use super::resource::{McpResource, McpResourceOutput};
use super::result::extract_tool_result;
use super::tool::McpTool;
use super::transport::{McpTransport, McpTransportType, UrlScrubber, connect_transport, secs};
use crate::error::{Error, Result};

/// How long [`McpClient::close`] waits for the session to shut down.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(10);

/// Most pages a list request may span before it is treated as runaway
/// pagination.
const MAX_PAGES: usize = 1000;

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
///
/// Every request (each page of a list, a tool call, a resource read, a
/// prompt render) is bounded by the transport's
/// [request timeout](McpTransport::with_request_timeout). A request that
/// times out is abandoned and returns an error; the session stays usable.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

struct Inner {
    peer: Peer<RoleClient>,
    info: Arc<ServerPeerInfo>,
    closed: AtomicBool,
    scrubber: UrlScrubber,
    request_timeout: Duration,
    service: tokio::sync::Mutex<Option<RunningService<RoleClient, ClientConfig>>>,
}

impl McpClient {
    /// Connects over `transport` and completes the initialize handshake.
    pub async fn connect(transport: McpTransport) -> Result<Self> {
        let scrubber = transport.scrubber();
        let request_timeout = transport.request_timeout();
        let service = connect_transport(&transport).await?;
        let info = service.peer_info().ok_or_else(|| {
            Error::Mcp("MCP server did not complete the initialize handshake".to_string())
        })?;
        Ok(Self {
            inner: Arc::new(Inner {
                peer: service.peer().clone(),
                info,
                closed: AtomicBool::new(false),
                scrubber,
                request_timeout,
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
        let tools = collect_pages("tools/list", self.pager(), |cursor| async move {
            let page = peer.list_tools(page_params(cursor)).await?;
            Ok((page.tools, page.next_cursor))
        })
        .await?;
        Ok(tools.into_iter().map(McpToolDefinition::from).collect())
    }

    /// Lists every tool as an [`McpTool`] bound to this session.
    pub async fn tools(&self) -> Result<Vec<McpTool>> {
        Ok(self
            .list_tools()
            .await?
            .into_iter()
            .map(|def| McpTool::from_definition(def, self.clone()))
            .collect())
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
        let resources = collect_pages("resources/list", self.pager(), |cursor| async move {
            let page = peer.list_resources(page_params(cursor)).await?;
            Ok((page.resources, page.next_cursor))
        })
        .await?;
        let templates = collect_pages(
            "resources/templates/list",
            self.pager(),
            |cursor| async move {
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
            },
        )
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

    /// Lists every resource and template as an [`McpResource`] bound to this
    /// session.
    pub async fn resources(&self) -> Result<Vec<McpResource>> {
        Ok(self
            .list_resources()
            .await?
            .into_iter()
            .map(|def| McpResource::from_definition(def, self.clone()))
            .collect())
    }

    /// Reads the concrete resource `uri` (`resources/read`).
    ///
    /// `uri` is sent verbatim; to read a template, fill it first with
    /// [`McpResource::uri_for`].
    pub async fn read_resource(&self, uri: &str) -> Result<McpResourceOutput> {
        let peer = self.peer()?;
        let result = self
            .request(
                "resources/read",
                uri,
                peer.read_resource_once(ReadResourceRequestParams::new(uri)),
            )
            .await?;
        McpResourceOutput::from_contents(result.contents)
    }

    /// Lists every prompt, following pagination. Empty when the server does
    /// not advertise the `prompts` capability.
    pub async fn list_prompts(&self) -> Result<Vec<McpPromptDefinition>> {
        let peer = self.peer()?;
        if self.server_capabilities().prompts.is_none() {
            return Ok(Vec::new());
        }
        let prompts = collect_pages("prompts/list", self.pager(), |cursor| async move {
            let page = peer.list_prompts(page_params(cursor)).await?;
            Ok((page.prompts, page.next_cursor))
        })
        .await?;
        Ok(prompts.into_iter().map(McpPromptDefinition::from).collect())
    }

    /// Lists every prompt as an [`McpPrompt`] bound to this session.
    pub async fn prompts(&self) -> Result<Vec<McpPrompt>> {
        Ok(self
            .list_prompts()
            .await?
            .into_iter()
            .map(|def| McpPrompt::from_definition(def, self.clone()))
            .collect())
    }

    /// Renders prompt `name` with `args` (`prompts/get`).
    pub async fn get_prompt(
        &self,
        name: &str,
        args: Map<String, Value>,
    ) -> Result<McpPromptOutput> {
        let peer = self.peer()?;
        let result = self
            .request(
                "prompts/get",
                name,
                peer.get_prompt_once(GetPromptRequestParams::new(name).with_arguments(args)),
            )
            .await?;
        McpPromptOutput::from_result(result)
    }

    /// Calls tool `name` with `args` and converts the result to JSON.
    ///
    /// A non-null `structuredContent` is returned as-is; a single text block
    /// is parsed as JSON when possible and otherwise returned as a string;
    /// any other content is returned as the MCP content array. A result with
    /// `isError: true` and JSON-RPC errors (such as an unknown tool) are `Err`.
    pub async fn call_tool(&self, name: &str, args: Map<String, Value>) -> Result<Value> {
        let peer = self.peer()?;
        let result = self
            .request(
                "tools/call",
                name,
                peer.call_tool_once(
                    CallToolRequestParams::new(name.to_string()).with_arguments(args),
                ),
            )
            .await?;
        extract_tool_result(name, result)
    }

    /// Pagination settings for this session.
    fn pager(&self) -> Pager<'_> {
        Pager {
            scrubber: &self.inner.scrubber,
            timeout: self.inner.request_timeout,
        }
    }

    /// Awaits a `*_once` request within the request timeout and turns its
    /// reply into the completed result, scrubbing any error text. On timeout
    /// the request future is dropped, which makes rmcp abandon (cancel) it.
    async fn request<R: Into<Reply<T>>, T>(
        &self,
        method: &str,
        subject: &str,
        request: impl Future<Output = std::result::Result<R, ServiceError>>,
    ) -> Result<T> {
        let timeout = self.inner.request_timeout;
        let reply = tokio::time::timeout(timeout, request).await.map_err(|_| {
            Error::Mcp(self.inner.scrubber.scrub(&format!(
                "{method} '{subject}' timed out after {}",
                secs(timeout)
            )))
        })?;
        reply
            .map_err(|e| format!("{method} '{subject}' failed: {e}"))
            .and_then(|r| r.into().into_complete(method, subject))
            .map_err(|msg| Error::Mcp(self.inner.scrubber.scrub(&msg)))
    }

    /// Gracefully ends the session for every clone. Idempotent; afterwards
    /// every request returns an error.
    pub async fn close(&self) -> Result<()> {
        self.inner.closed.store(true, Ordering::SeqCst);
        let service = self.inner.service.lock().await.take();
        let Some(mut service) = service else {
            return Ok(());
        };
        match service.close_with_timeout(CLOSE_TIMEOUT).await {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err(Error::Mcp("timed out closing MCP session".to_string())),
            Err(e) => Err(Error::Mcp(
                self.inner
                    .scrubber
                    .scrub(&format!("failed to close MCP session: {e}")),
            )),
        }
    }
}

/// A reply to `tools/call`, `resources/read` or `prompts/get`, reduced to the
/// cases this client can act on. rmcp's `*_once` calls return these
/// without driving the multi round-trip (MRTR) follow-ups; the plain calls
/// would turn a non-complete reply into an opaque "Unexpected response type".
enum Reply<T> {
    Complete(T),
    InputRequired,
    Task,
    /// A variant added to rmcp's `#[non_exhaustive]` response enums.
    Other,
}

impl<T> Reply<T> {
    /// The completed result, or the (unscrubbed) error message for any other
    /// reply.
    fn into_complete(self, method: &str, subject: &str) -> std::result::Result<T, String> {
        match self {
            Self::Complete(t) => Ok(t),
            Self::InputRequired => Err(format!(
                "{method} '{subject}' failed: server requested additional input (input_required), which this client does not support"
            )),
            Self::Task => Err(format!(
                "{method} '{subject}' failed: server returned a task, which this client does not support"
            )),
            Self::Other => Err(format!(
                "{method} '{subject}' failed: server returned an unsupported response type"
            )),
        }
    }
}

impl From<CallToolResponse> for Reply<rmcp::model::CallToolResult> {
    fn from(r: CallToolResponse) -> Self {
        match r {
            CallToolResponse::Complete(t) => Self::Complete(t),
            CallToolResponse::InputRequired(_) => Self::InputRequired,
            CallToolResponse::Task(_) => Self::Task,
            _ => Self::Other,
        }
    }
}

impl From<ReadResourceResponse> for Reply<rmcp::model::ReadResourceResult> {
    fn from(r: ReadResourceResponse) -> Self {
        match r {
            ReadResourceResponse::Complete(t) => Self::Complete(t),
            ReadResourceResponse::InputRequired(_) => Self::InputRequired,
            _ => Self::Other,
        }
    }
}

impl From<GetPromptResponse> for Reply<rmcp::model::GetPromptResult> {
    fn from(r: GetPromptResponse) -> Self {
        match r {
            GetPromptResponse::Complete(t) => Self::Complete(t),
            GetPromptResponse::InputRequired(_) => Self::InputRequired,
            _ => Self::Other,
        }
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

/// How [`collect_pages`] scrubs errors and bounds each page request.
#[derive(Clone, Copy)]
pub(crate) struct Pager<'a> {
    pub(crate) scrubber: &'a UrlScrubber,
    pub(crate) timeout: Duration,
}

/// Follows opaque `nextCursor` values, concatenating every page's items.
///
/// rmcp's `list_all_*` helpers do not detect cycles, so this loop stops with
/// [`McpPaginationError::RepeatedCursor`] when a cursor repeats, and with an
/// error after [`MAX_PAGES`] pages, so a misbehaving server cannot cause an
/// infinite loop or unbounded memory growth. Each page is bounded by
/// `pager.timeout`. Errors from `fetch` (including continuation-page
/// failures) are returned with `method` as context.
pub(crate) async fn collect_pages<T, F, Fut>(
    method: &str,
    pager: Pager<'_>,
    mut fetch: F,
) -> Result<Vec<T>>
where
    F: FnMut(Option<String>) -> Fut,
    Fut: Future<Output = std::result::Result<(Vec<T>, Option<String>), ServiceError>>,
{
    let scrubber = pager.scrubber;
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let mut cursor = None;
    for _ in 0..MAX_PAGES {
        let (page, next) = tokio::time::timeout(pager.timeout, fetch(cursor.take()))
            .await
            .map_err(|_| Error::Mcp(format!("{method} timed out after {}", secs(pager.timeout))))?
            .map_err(|e| Error::Mcp(scrubber.scrub(&format!("{method} failed: {e}"))))?;
        items.extend(page);
        match next {
            Some(next) => {
                if !seen.insert(next.clone()) {
                    return Err(Error::Mcp(format!(
                        "{method} failed: {}",
                        McpPaginationError::RepeatedCursor(next)
                    )));
                }
                cursor = Some(next);
            }
            None => return Ok(items),
        }
    }
    Err(Error::Mcp(format!(
        "{method} failed: more than {MAX_PAGES} pages"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    type Page = std::result::Result<(Vec<u32>, Option<String>), ServiceError>;

    static NO_SCRUB: std::sync::LazyLock<UrlScrubber> = std::sync::LazyLock::new(UrlScrubber::none);

    fn test_pager() -> Pager<'static> {
        Pager {
            scrubber: &NO_SCRUB,
            timeout: Duration::from_secs(5),
        }
    }

    #[tokio::test]
    async fn page_count_is_capped() {
        let fetched = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = fetched.clone();
        // A fresh cursor every time, so only the page cap can stop it.
        let fetch = move |_cursor| {
            let n = count.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok((vec![n as u32], Some(format!("c{n}")))))
        };
        let err = collect_pages("x/list", test_pager(), fetch)
            .await
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "MCP error: x/list failed: more than 1000 pages"
        );
        assert_eq!(fetched.load(Ordering::SeqCst), 1000);
    }

    #[tokio::test]
    async fn a_page_under_the_cap_is_fine() {
        let fetched = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = fetched.clone();
        let fetch = move |_cursor| {
            let n = count.fetch_add(1, Ordering::SeqCst);
            let next = (n + 1 < 1000).then(|| format!("c{n}"));
            std::future::ready(Ok((vec![n as u32], next)))
        };
        let items = collect_pages("x/list", test_pager(), fetch).await.unwrap();
        assert_eq!(items.len(), 1000);
    }

    #[tokio::test(start_paused = true)]
    async fn each_page_is_bounded_by_the_timeout() {
        let fetch = |_cursor| async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok::<_, ServiceError>((Vec::<u32>::new(), None))
        };
        let pager = Pager {
            scrubber: &NO_SCRUB,
            timeout: Duration::from_millis(1500),
        };
        let err = collect_pages("x/list", pager, fetch).await.unwrap_err();
        assert_eq!(err.to_string(), "MCP error: x/list timed out after 1.5s");
    }
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
        assert_eq!(
            collect_pages("x/list", test_pager(), fetch).await.unwrap(),
            [1, 2, 3, 4]
        );
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
        let err = collect_pages("x/list", test_pager(), fetch)
            .await
            .unwrap_err();
        let inner = McpPaginationError::RepeatedCursor("c1".into());
        assert!(
            err.to_string().contains(&format!("x/list failed: {inner}")),
            "{err}"
        );
        assert!(err.to_string().contains("repeated cursor"), "{err}");
    }

    #[tokio::test]
    async fn continuation_error_propagates() {
        let (_, fetch) = scripted(vec![
            Ok((vec![1], Some("c1".into()))),
            Err(ServiceError::TransportClosed),
        ]);
        let err = collect_pages("x/list", test_pager(), fetch)
            .await
            .unwrap_err();
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

#[cfg(test)]
mod reply_tests {
    use super::*;

    #[test]
    fn complete_passes_through() {
        assert_eq!(Reply::Complete(7).into_complete("m", "s"), Ok(7));
    }

    #[test]
    fn non_complete_replies_are_explicit_errors() {
        let err = Reply::<()>::InputRequired
            .into_complete("prompts/get", "p")
            .unwrap_err();
        assert_eq!(
            err,
            "prompts/get 'p' failed: server requested additional input (input_required), which this client does not support"
        );
        let err = Reply::<()>::Task
            .into_complete("tools/call", "t")
            .unwrap_err();
        assert!(err.contains("server returned a task"), "{err}");
        let err = Reply::<()>::Other
            .into_complete("tools/call", "t")
            .unwrap_err();
        assert!(err.contains("unsupported response type"), "{err}");
    }
}
