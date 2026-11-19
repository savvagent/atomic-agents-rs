use std::time::Duration;

use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Which kind of results to return.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FirecrawlSearchType {
    /// Web pages.
    #[default]
    Web,
    /// News articles.
    News,
}

impl FirecrawlSearchType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::News => "news",
        }
    }
}

/// Time window for web results.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FirecrawlTimeRange {
    /// Past day.
    Day,
    /// Past week.
    Week,
    /// Past month.
    Month,
    /// Past year.
    Year,
}

impl FirecrawlTimeRange {
    fn tbs(self) -> &'static str {
        match self {
            Self::Day => "qdr:d",
            Self::Week => "qdr:w",
            Self::Month => "qdr:m",
            Self::Year => "qdr:y",
        }
    }
}

fn default_max_results() -> u32 {
    5
}

/// Search the web or recent news through the Firecrawl Search API.
/// Use this to find current information on a topic. Set include_content to true to also get
/// each result page as Markdown in the same call. Requires a Firecrawl API key.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct FirecrawlSearchInput {
    /// Search queries to run. Supports operators such as site:, "exact phrase",
    /// -term, and filetype:pdf.
    pub queries: Vec<String>,
    /// Which results to return. 'web' = web pages; 'news' = news articles.
    #[serde(default)]
    pub search_type: FirecrawlSearchType,
    /// Maximum results per query (1-100).
    #[serde(default = "default_max_results")]
    pub max_results_per_query: u32,
    /// Also return the main content of each result page as Markdown. Slower than
    /// a plain search.
    #[serde(default)]
    pub include_content: bool,
}

/// A single Firecrawl search result (web page or news article).
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct FirecrawlSearchResultItem {
    /// The query that produced this result.
    pub query: String,
    /// Title of the result.
    pub title: String,
    /// URL of the result.
    pub url: String,
    /// Query-relevant excerpt or snippet for the result.
    pub description: Option<String>,
    /// Rank of the result.
    pub position: Option<u32>,
    /// Publication date string (news only).
    pub published: Option<String>,
    /// Page content as Markdown (only when include_content is set).
    pub content: Option<String>,
}

/// A query that failed while other queries returned results.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct FirecrawlSearchFailure {
    /// The query that failed.
    pub query: String,
    /// Why it failed, e.g. an HTTP status and the API's error message.
    pub error: String,
}

/// Output of the Firecrawl search tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct FirecrawlSearchOutput {
    /// Matching results across all queries.
    pub results: Vec<FirecrawlSearchResultItem>,
    /// Queries that failed while others returned results, with the reason.
    #[serde(default)]
    pub failures: Vec<FirecrawlSearchFailure>,
}

/// Configuration for the Firecrawl search tool.
pub struct FirecrawlSearchConfig {
    pub component: ComponentConfig,
    /// Firecrawl API key. Falls back to the `FIRECRAWL_API_KEY` environment variable when empty.
    pub api_key: String,
    /// Firecrawl API base URL.
    pub base_url: String,
    /// Location to localise results, e.g. "Germany" or "San Francisco,California,United States".
    pub location: Option<String>,
    /// Only return web results from this time window. Does not apply to news.
    pub time_range: Option<FirecrawlTimeRange>,
    /// Truncate each result's content to this many characters. No limit when unset.
    pub max_content_chars: Option<usize>,
    /// HTTP request timeout.
    pub timeout: Duration,
}

impl FirecrawlSearchConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("FirecrawlSearch")
                .with_description("Search the web or news using Firecrawl"),
            api_key: api_key.into(),
            base_url: "https://api.firecrawl.dev/v2".to_string(),
            location: None,
            time_range: None,
            max_content_chars: None,
            timeout: Duration::from_secs(75),
        }
    }
}

/// Tool for searching the web and news via the Firecrawl Search API.
///
/// # Example
///
/// ```no_run
/// use atomic_agents::prelude::*;
/// use atomic_forge::firecrawl_search::*;
///
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let tool = FirecrawlSearchTool::new(FirecrawlSearchConfig::new("fc-..."));
/// let out = tool.run(serde_json::from_str(r#"{"queries": ["rust async"]}"#).unwrap()).unwrap();
/// println!("{}", out.results.len());
/// # }
/// ```
pub struct FirecrawlSearchTool {
    config: ComponentConfig,
    api_key: String,
    base_url: String,
    location: Option<String>,
    time_range: Option<FirecrawlTimeRange>,
    max_content_chars: Option<usize>,
    client: reqwest::Client,
}

const MIN_REDACTED_KEY_LENGTH: usize = 8;

/// Replace every case-insensitive occurrence of `needle` in `text`.
fn replace_ignore_ascii_case(text: &str, needle: &str, with: &str) -> String {
    if needle.is_empty() {
        return text.to_string();
    }
    let hay = text.to_ascii_lowercase();
    let needle = needle.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    while let Some(pos) = hay[last..].find(&needle) {
        let start = last + pos;
        out.push_str(&text[last..start]);
        out.push_str(with);
        last = start + needle.len();
    }
    out.push_str(&text[last..]);
    out
}

/// Redact anything shaped like a Firecrawl key (`fc-` + 32 hex digits, on word boundaries).
fn redact_key_pattern(text: &str) -> String {
    let bytes = text.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let is_key = i + 35 <= bytes.len()
            && bytes[i..i + 3].eq_ignore_ascii_case(b"fc-")
            && bytes[i + 3..i + 35].iter().all(u8::is_ascii_hexdigit)
            && (i == 0 || !is_word(bytes[i - 1]))
            && (i + 35 == bytes.len() || !is_word(bytes[i + 35]));
        if is_key {
            out.push_str("[redacted]");
            i += 35;
        } else {
            // Copy one whole character.
            let ch = text[i..].chars().next().expect("index is on a char boundary");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn first_text(hit: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|k| hit.get(*k).and_then(Value::as_str))
        .find(|s| !s.is_empty())
        .map(String::from)
}

impl FirecrawlSearchTool {
    pub fn new(config: FirecrawlSearchConfig) -> Self {
        let api_key = if config.api_key.is_empty() {
            std::env::var("FIRECRAWL_API_KEY").unwrap_or_default()
        } else {
            config.api_key
        };
        // Never follow redirects: the Authorization header must not leave the API host.
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_default();
        Self {
            config: config.component,
            api_key,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            location: config.location,
            time_range: config.time_range,
            max_content_chars: config.max_content_chars,
            client,
        }
    }

    fn redact(&self, text: &str) -> String {
        let text = if self.api_key.len() >= MIN_REDACTED_KEY_LENGTH {
            replace_ignore_ascii_case(text, &self.api_key, "[redacted]")
        } else {
            text.to_string()
        };
        redact_key_pattern(&text)
    }

    /// True for a printable http(s) URL with a host and a non-zero port.
    ///
    /// ```
    /// use atomic_forge::firecrawl_search::FirecrawlSearchTool;
    /// assert!(FirecrawlSearchTool::usable_url("https://example.com/a?b=1"));
    /// assert!(!FirecrawlSearchTool::usable_url("javascript:alert(1)"));
    /// assert!(!FirecrawlSearchTool::usable_url("https://exa mple.com"));
    /// assert!(!FirecrawlSearchTool::usable_url("https://example.com:0/"));
    /// ```
    pub fn usable_url(raw: &str) -> bool {
        if raw.is_empty() || raw.chars().any(|c| c.is_whitespace() || c.is_control() || c == '\\') {
            return false;
        }
        match url::Url::parse(raw) {
            Ok(u) => {
                matches!(u.scheme(), "http" | "https")
                    && u.host_str().is_some_and(|h| !h.is_empty())
                    && u.port() != Some(0)
            }
            Err(_) => false,
        }
    }

    /// Normalise one result, or `None` when it has no usable http(s) URL.
    pub fn to_item(
        hit: &Value,
        query: &str,
        max_content_chars: Option<usize>,
    ) -> Option<FirecrawlSearchResultItem> {
        let url = hit.get("url").and_then(Value::as_str)?;
        if !Self::usable_url(url) {
            return None;
        }
        let content = first_text(hit, &["markdown"]).map(|c| match max_content_chars {
            Some(n) => c.chars().take(n).collect(),
            None => c,
        });
        Some(FirecrawlSearchResultItem {
            query: query.to_string(),
            title: first_text(hit, &["title"]).unwrap_or_else(|| url.to_string()),
            url: url.to_string(),
            description: first_text(hit, &["description", "snippet"]),
            position: hit
                .get("position")
                .and_then(Value::as_u64)
                .filter(|p| *p > 0)
                .and_then(|p| u32::try_from(p).ok()),
            published: first_text(hit, &["date"]),
            content,
        })
    }

    fn build_body(&self, query: &str, input: &FirecrawlSearchInput) -> Value {
        let mut body = serde_json::json!({
            "query": query,
            "limit": input.max_results_per_query.clamp(1, 100),
            "sources": [input.search_type.as_str()],
        });
        if let Some(loc) = self.location.as_deref().filter(|l| !l.is_empty()) {
            body["location"] = Value::from(loc);
        }
        if let Some(range) = self.time_range
            && input.search_type == FirecrawlSearchType::Web
        {
            body["tbs"] = Value::from(range.tbs());
        }
        if input.include_content {
            body["scrapeOptions"] = serde_json::json!({"formats": ["markdown"], "onlyMainContent": true});
        }
        body
    }

    /// Parse a successful response body into result items, or say why it cannot be used.
    pub fn parse_response(
        payload: &Value,
        query: &str,
        search_type: FirecrawlSearchType,
        max_content_chars: Option<usize>,
    ) -> std::result::Result<Vec<FirecrawlSearchResultItem>, String> {
        // The API leaves the source key out when there are no results.
        let hits = match payload.get("data") {
            Some(Value::Object(groups)) => match groups.get(search_type.as_str()) {
                None => Some(&[][..]),
                Some(Value::Array(hits)) => Some(hits.as_slice()),
                Some(_) => None,
            },
            _ => None,
        }
        .ok_or_else(|| {
            format!(
                "200 unexpected response shape for '{}' results",
                search_type.as_str()
            )
        })?;
        let items: Vec<_> = hits
            .iter()
            .filter_map(|h| Self::to_item(h, query, max_content_chars))
            .collect();
        if !hits.is_empty() && items.is_empty() {
            return Err(format!(
                "200 none of the {} '{}' results had a usable url",
                hits.len(),
                search_type.as_str()
            ));
        }
        Ok(items)
    }

    async fn fetch(
        &self,
        query: &str,
        input: &FirecrawlSearchInput,
    ) -> std::result::Result<Vec<FirecrawlSearchResultItem>, String> {
        let fail = |detail: String| self.redact(&format!("Firecrawl search failed for '{query}': {detail}"));
        let response = self
            .client
            .post(format!("{}/search", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&self.build_body(query, input))
            .send()
            .await
            .map_err(|e| fail(format!("request failed: {}", e.without_url())))?;
        let status = response.status();
        let payload: Value = response.json().await.unwrap_or(Value::Null);
        let succeeded = status == reqwest::StatusCode::OK && payload.get("success") == Some(&Value::Bool(true));
        if !succeeded {
            let reason = payload
                .get("error")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .unwrap_or_else(|| {
                    if status == reqwest::StatusCode::OK {
                        "response did not report success".to_string()
                    } else {
                        status.canonical_reason().unwrap_or("").to_string()
                    }
                });
            return Err(fail(format!("{} {reason}", status.as_u16())));
        }
        let mut items = Self::parse_response(&payload, query, input.search_type, self.max_content_chars)
            .map_err(fail)?;
        items.truncate(input.max_results_per_query.clamp(1, 100) as usize);
        Ok(items)
    }
}

impl Tool for FirecrawlSearchTool {
    type Input = FirecrawlSearchInput;
    type Output = FirecrawlSearchOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        if self.api_key.is_empty() {
            return Err(Error::Config(
                "Firecrawl API key is missing. Set FirecrawlSearchConfig::api_key or the \
                 FIRECRAWL_API_KEY environment variable."
                    .to_string(),
            ));
        }
        // Duplicate queries would be billed twice and return the same results.
        let mut queries: Vec<&String> = Vec::new();
        for q in &input.queries {
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        if queries.is_empty() {
            return Err(Error::Validation("At least one query is required.".to_string()));
        }

        let groups = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(futures::future::join_all(
                queries.iter().map(|q| self.fetch(q, &input)),
            ))
        });

        let mut results = Vec::new();
        let mut failures: Vec<FirecrawlSearchFailure> = Vec::new();
        for (query, group) in queries.iter().zip(groups) {
            match group {
                Ok(items) => results.extend(items),
                Err(error) => {
                    tracing::warn!("{error}");
                    failures.push(FirecrawlSearchFailure {
                        query: self.redact(query),
                        error,
                    });
                }
            }
        }
        // A bad key, exhausted credits or a malformed response must never look like an empty search.
        if results.is_empty()
            && let Some(first) = failures.first()
        {
            return Err(Error::Validation(first.error.clone()));
        }
        Ok(FirecrawlSearchOutput { results, failures })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const KEY: &str = "fc-0123456789abcdef0123456789abcdef";

    fn tool_for(uri: &str) -> FirecrawlSearchTool {
        let mut cfg = FirecrawlSearchConfig::new(KEY);
        cfg.base_url = uri.to_string();
        FirecrawlSearchTool::new(cfg)
    }

    fn input(queries: &[&str]) -> FirecrawlSearchInput {
        serde_json::from_value(json!({ "queries": queries })).unwrap()
    }

    #[test]
    fn schema_metadata() {
        assert!(FirecrawlSearchInput::schema_description().contains("Firecrawl"));
        let schema = serde_json::to_value(FirecrawlSearchInput::generate_schema()).unwrap();
        assert!(schema["properties"].get("include_content").is_some());
        let out = serde_json::to_value(FirecrawlSearchOutput::generate_schema()).unwrap();
        assert!(out["properties"].get("failures").is_some());
    }

    #[test]
    fn defaults_apply() {
        let i = input(&["a"]);
        assert_eq!(i.search_type, FirecrawlSearchType::Web);
        assert_eq!(i.max_results_per_query, 5);
        assert!(!i.include_content);
        assert!(serde_json::from_value::<FirecrawlSearchInput>(
            json!({"queries": ["a"], "search_type": "images"})
        )
        .is_err());
    }

    #[test]
    fn url_validation() {
        assert!(FirecrawlSearchTool::usable_url("http://a.com:8080/x#y"));
        assert!(!FirecrawlSearchTool::usable_url(""));
        assert!(!FirecrawlSearchTool::usable_url("ftp://a.com"));
        assert!(!FirecrawlSearchTool::usable_url("https://a.com\\@b.com"));
        assert!(!FirecrawlSearchTool::usable_url("https://a.com/\nx"));
        assert!(!FirecrawlSearchTool::usable_url("https://"));
        assert!(!FirecrawlSearchTool::usable_url("https://a.com:99999/"));
    }

    #[test]
    fn to_item_normalises_fields() {
        let hit = json!({
            "url": "https://a.com", "title": "", "snippet": "snip", "position": 2,
            "date": "2 days ago", "markdown": "abcdef"
        });
        let item = FirecrawlSearchTool::to_item(&hit, "q", Some(3)).unwrap();
        assert_eq!(item.title, "https://a.com");
        assert_eq!(item.description.as_deref(), Some("snip"));
        assert_eq!(item.position, Some(2));
        assert_eq!(item.published.as_deref(), Some("2 days ago"));
        assert_eq!(item.content.as_deref(), Some("abc"));

        for bad in [json!(true), json!(0), json!(1.5), json!("1"), json!(-3)] {
            let hit = json!({"url": "https://a.com", "position": bad});
            assert_eq!(FirecrawlSearchTool::to_item(&hit, "q", None).unwrap().position, None);
        }
        assert!(FirecrawlSearchTool::to_item(&json!({"title": "x"}), "q", None).is_none());
    }

    #[test]
    fn parse_response_shapes() {
        let t = FirecrawlSearchType::Web;
        let ok = json!({"success": true, "data": {"web": [{"url": "https://a.com"}]}});
        assert_eq!(FirecrawlSearchTool::parse_response(&ok, "q", t, None).unwrap().len(), 1);
        let empty = json!({"success": true, "data": {}});
        assert!(FirecrawlSearchTool::parse_response(&empty, "q", t, None).unwrap().is_empty());
        let bad_shape = json!({"success": true, "data": []});
        assert!(FirecrawlSearchTool::parse_response(&bad_shape, "q", t, None)
            .unwrap_err()
            .contains("unexpected response shape"));
        let unusable = json!({"data": {"web": [{"url": "nope"}]}});
        assert!(FirecrawlSearchTool::parse_response(&unusable, "q", t, None)
            .unwrap_err()
            .contains("usable url"));
    }

    #[test]
    fn redaction() {
        let tool = tool_for("http://x");
        let text = format!("bad key {KEY} and {} again", KEY.to_uppercase());
        assert!(!tool.redact(&text).contains("0123456789abcdef"));
        // A different key in the right shape is redacted too.
        let other = "fc-ffffffffffffffffffffffffffffffff";
        assert_eq!(redact_key_pattern(other), "[redacted]");
        assert_eq!(redact_key_pattern("xfc-ffffffffffffffffffffffffffffffff"), "xfc-ffffffffffffffffffffffffffffffff");
        assert_eq!(redact_key_pattern("ünï fc-short"), "ünï fc-short");
    }

    #[test]
    fn request_body() {
        let mut cfg = FirecrawlSearchConfig::new(KEY);
        cfg.location = Some("Germany".into());
        cfg.time_range = Some(FirecrawlTimeRange::Week);
        let tool = FirecrawlSearchTool::new(cfg);
        let mut i = input(&["a"]);
        i.include_content = true;
        let body = tool.build_body("a", &i);
        assert_eq!(body["sources"], json!(["web"]));
        assert_eq!(body["tbs"], "qdr:w");
        assert_eq!(body["location"], "Germany");
        assert_eq!(body["scrapeOptions"]["formats"], json!(["markdown"]));
        i.search_type = FirecrawlSearchType::News;
        assert!(tool.build_body("a", &i).get("tbs").is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn missing_key_is_config_error() {
        // SAFETY: tests in this crate do not rely on FIRECRAWL_API_KEY.
        unsafe { std::env::remove_var("FIRECRAWL_API_KEY") };
        let tool = FirecrawlSearchTool::new(FirecrawlSearchConfig::new(""));
        assert!(matches!(tool.run(input(&["a"])), Err(Error::Config(_))));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_queries_is_error() {
        let tool = tool_for("http://x");
        assert!(matches!(tool.run(input(&[])), Err(Error::Validation(_))));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_parses_canned_response_and_dedupes() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .and(header("authorization", format!("Bearer {KEY}").as_str()))
            .and(body_partial_json(json!({"query": "rust", "sources": ["news"]})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "success": true,
                "data": {"news": [
                    {"title": "T1", "url": "https://n.com/1", "snippet": "s", "date": "today", "position": 1},
                    {"title": "bad", "url": "javascript:x"}
                ]}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        let mut i = input(&["rust", "rust"]);
        i.search_type = FirecrawlSearchType::News;
        let out = tool.run(i).unwrap();
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.results[0].title, "T1");
        assert!(out.failures.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn partial_failure_is_reported() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"query": "good"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "success": true, "data": {"web": [{"url": "https://a.com", "title": "A"}]}
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"query": "bad"})))
            .respond_with(ResponseTemplate::new(402).set_body_json(json!({
                "success": false, "error": format!("Out of credits for {KEY}")
            })))
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        let out = tool.run(input(&["good", "bad"])).unwrap();
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.failures.len(), 1);
        assert_eq!(out.failures[0].query, "bad");
        assert!(out.failures[0].error.contains("402 Out of credits"));
        assert!(!out.failures[0].error.contains("0123456789abcdef"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn all_failures_raise() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({"error": "Unauthorized"})))
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        match tool.run(input(&["a"])) {
            Err(Error::Validation(msg)) => assert!(msg.contains("401 Unauthorized"), "{msg}"),
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn success_false_on_200_is_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {"web": []}})))
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        assert!(tool.run(input(&["a"])).is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires network access and FIRECRAWL_API_KEY"]
    async fn live_search() {
        let tool = FirecrawlSearchTool::new(FirecrawlSearchConfig::new(""));
        let out = tool.run(input(&["rust programming language"])).unwrap();
        assert!(!out.results.is_empty());
    }
}
