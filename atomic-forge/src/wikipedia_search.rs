use std::time::Duration;

use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_language() -> String {
    "en".to_string()
}

fn default_max_results() -> u32 {
    3
}

/// Schema for searching Wikipedia. Returns matching articles with titles, URLs,
/// and plain-text summaries. Use this whenever you need encyclopedic background on a
/// person, place, concept, or event. Free public API; no key required.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct WikipediaSearchInput {
    /// One or more search queries to look up on Wikipedia.
    pub queries: Vec<String>,
    /// Wikipedia edition code as a single hostname label (e.g. 'en', 'fr',
    /// 'simple', 'zh-min-nan').
    #[serde(default = "default_language")]
    pub language: String,
    /// Maximum number of articles to return per query (1-20).
    #[serde(default = "default_max_results")]
    pub max_results_per_query: u32,
    /// When true, fetch the full plain-text extract for the top returned result
    /// of each query in addition to the short summary. Costs an extra API call
    /// per query, so leave this off unless you need full article content.
    #[serde(default)]
    pub full_text: bool,
}

/// A single Wikipedia article result.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct WikipediaArticle {
    /// The query that produced this result.
    pub query: String,
    /// The article title.
    pub title: String,
    /// Canonical URL of the article on Wikipedia.
    pub page_url: String,
    /// Plain-text lead summary of the article.
    pub summary: String,
    /// Short description (one-liner) for the article, when available.
    pub description: Option<String>,
    /// Full plain-text article content, populated only when `full_text` was requested.
    pub full_text: Option<String>,
    /// URL of a thumbnail image for the article, when available.
    pub thumbnail_url: Option<String>,
}

/// Output of the Wikipedia search tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct WikipediaSearchOutput {
    /// Articles returned across all queries, in best-match order per query.
    pub results: Vec<WikipediaArticle>,
}

/// Configuration for the Wikipedia search tool.
pub struct WikipediaSearchConfig {
    pub component: ComponentConfig,
    /// Base URL template; `{lang}` is replaced with the (validated) language code.
    pub base_url_template: String,
    /// User agent for Wikipedia API requests. Wikipedia asks for a descriptive UA.
    pub user_agent: String,
    /// HTTP request timeout.
    pub timeout: Duration,
}

impl Default for WikipediaSearchConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("WikipediaSearch")
                .with_description("Search Wikipedia articles"),
            base_url_template: "https://{lang}.wikipedia.org".to_string(),
            user_agent: "atomic-agents-wikipedia-tool/1.0 (+https://github.com/savvagent/atomic-agents-rs)"
                .to_string(),
            timeout: Duration::from_secs(15),
        }
    }
}

/// Tool for searching Wikipedia articles via the public MediaWiki and REST APIs.
///
/// # Example
///
/// ```no_run
/// use atomic_agents::prelude::*;
/// use atomic_forge::wikipedia_search::*;
///
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let tool = WikipediaSearchTool::new(WikipediaSearchConfig::default());
/// let out = tool.run(serde_json::from_str(r#"{"queries": ["Rust (programming language)"]}"#).unwrap()).unwrap();
/// println!("{}", out.results.len());
/// # }
/// ```
pub struct WikipediaSearchTool {
    config: ComponentConfig,
    base_url_template: String,
    client: reqwest::Client,
}

/// Validate a Wikipedia edition code: a single hostname label, 1-63 chars.
///
/// ```
/// use atomic_forge::wikipedia_search::is_valid_language;
/// assert!(is_valid_language("zh-min-nan"));
/// assert!(!is_valid_language("en.evil.com"));
/// ```
pub fn is_valid_language(lang: &str) -> bool {
    let bytes = lang.as_bytes();
    if bytes.is_empty() || bytes.len() > 63 {
        return false;
    }
    let alnum = |b: &u8| b.is_ascii_alphanumeric();
    alnum(&bytes[0])
        && alnum(&bytes[bytes.len() - 1])
        && bytes.iter().all(|b| alnum(b) || *b == b'-')
}

impl WikipediaSearchTool {
    pub fn new(config: WikipediaSearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(config.user_agent)
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self {
            config: config.component,
            base_url_template: config.base_url_template,
            client,
        }
    }

    fn api_base(&self, language: &str) -> String {
        self.base_url_template
            .replace("{lang}", language)
            .trim_end_matches('/')
            .to_string()
    }

    async fn search_titles(
        &self,
        query: &str,
        language: &str,
        limit: u32,
    ) -> std::result::Result<Vec<String>, Error> {
        let params = [
            ("action", "query".to_string()),
            ("list", "search".to_string()),
            ("srsearch", query.to_string()),
            ("srlimit", limit.to_string()),
            ("format", "json".to_string()),
            ("utf8", String::new()),
        ];
        let response = self
            .client
            .get(format!("{}/w/api.php", self.api_base(language)))
            .query(&params)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("Wikipedia request failed: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Validation(format!(
                "Wikipedia search failed for '{query}': {status}"
            )));
        }
        let data: Value = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse Wikipedia response: {e}")))?;
        Ok(Self::parse_titles(&data))
    }

    /// Extract article titles from a MediaWiki `list=search` response.
    pub fn parse_titles(data: &Value) -> Vec<String> {
        data["query"]["search"]
            .as_array()
            .map(|hits| {
                hits.iter()
                    .filter_map(|h| h.get("title").and_then(Value::as_str))
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn fetch_summary(
        &self,
        title: &str,
        language: &str,
    ) -> std::result::Result<Option<Value>, Error> {
        let mut url = url::Url::parse(&self.api_base(language))
            .map_err(|e| Error::Config(format!("Invalid Wikipedia base URL: {e}")))?;
        url.path_segments_mut()
            .map_err(|_| Error::Config("Wikipedia base URL cannot be a base".to_string()))?
            .pop_if_empty()
            .extend(["api", "rest_v1", "page", "summary", title]);
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("Wikipedia request failed: {e}")))?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(Error::Validation(format!(
                "Wikipedia summary failed for '{title}': {status}"
            )));
        }
        let data = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse Wikipedia summary: {e}")))?;
        Ok(Some(data))
    }

    async fn fetch_full_extract(
        &self,
        title: &str,
        language: &str,
    ) -> std::result::Result<Option<String>, Error> {
        let params = [
            ("action", "query"),
            ("prop", "extracts"),
            ("explaintext", "true"),
            ("titles", title),
            ("format", "json"),
            ("redirects", "1"),
            ("utf8", ""),
        ];
        let response = self
            .client
            .get(format!("{}/w/api.php", self.api_base(language)))
            .query(&params)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("Wikipedia request failed: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Validation(format!(
                "Wikipedia extract failed for '{title}': {status}"
            )));
        }
        let data: Value = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse Wikipedia extract: {e}")))?;
        Ok(data["query"]["pages"].as_object().and_then(|pages| {
            pages
                .values()
                .filter_map(|p| p.get("extract").and_then(Value::as_str))
                .find(|s| !s.is_empty())
                .map(String::from)
        }))
    }

    /// Build an article from a REST summary payload.
    pub fn article_from_summary(
        &self,
        summary: &Value,
        query: &str,
        title: &str,
        language: &str,
        full_text: Option<String>,
    ) -> WikipediaArticle {
        let page_url = summary["content_urls"]["desktop"]["page"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| format!("{}/wiki/{}", self.api_base(language), title.replace(' ', "_")));
        WikipediaArticle {
            query: query.to_string(),
            title: summary
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or(title)
                .to_string(),
            page_url,
            summary: summary
                .get("extract")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            description: summary
                .get("description")
                .and_then(Value::as_str)
                .map(String::from),
            full_text,
            thumbnail_url: summary["thumbnail"]["source"].as_str().map(String::from),
        }
    }

    async fn process_query(
        &self,
        query: &str,
        input: &WikipediaSearchInput,
    ) -> std::result::Result<Vec<WikipediaArticle>, Error> {
        let language = input.language.as_str();
        let limit = input.max_results_per_query.clamp(1, 20);
        let titles = self.search_titles(query, language, limit).await?;
        let mut articles: Vec<WikipediaArticle> = Vec::new();
        for title in titles {
            let summary = match self.fetch_summary(&title, language).await {
                Ok(Some(s)) => s,
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!("Failed to fetch summary for '{title}': {e}");
                    continue;
                }
            };
            let mut full = None;
            if input.full_text && articles.is_empty() {
                match self.fetch_full_extract(&title, language).await {
                    Ok(f) => full = f,
                    Err(e) => tracing::warn!("Failed to fetch full extract for '{title}': {e}"),
                }
            }
            articles.push(self.article_from_summary(&summary, query, &title, language, full));
        }
        Ok(articles)
    }
}

impl Tool for WikipediaSearchTool {
    type Input = WikipediaSearchInput;
    type Output = WikipediaSearchOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        if !is_valid_language(&input.language) {
            return Err(Error::Validation(format!(
                "Invalid Wikipedia language code '{}': must be a single hostname label",
                input.language
            )));
        }
        let results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let groups = futures::future::join_all(
                    input.queries.iter().map(|q| self.process_query(q, &input)),
                )
                .await;
                let mut all = Vec::new();
                for (query, group) in input.queries.iter().zip(groups) {
                    match group {
                        Ok(articles) => all.extend(articles),
                        Err(e) => tracing::warn!("Wikipedia query '{query}' failed: {e}"),
                    }
                }
                all
            })
        });
        Ok(WikipediaSearchOutput { results })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn input(queries: &[&str]) -> WikipediaSearchInput {
        serde_json::from_value(json!({ "queries": queries })).unwrap()
    }

    fn tool_for(uri: String) -> WikipediaSearchTool {
        WikipediaSearchTool::new(WikipediaSearchConfig {
            base_url_template: uri,
            ..Default::default()
        })
    }

    #[test]
    fn schema_metadata() {
        assert!(WikipediaSearchInput::schema_description().contains("Wikipedia"));
        let schema = serde_json::to_value(WikipediaSearchInput::generate_schema()).unwrap();
        assert!(schema["properties"].get("full_text").is_some());
        let out = serde_json::to_value(WikipediaArticle::generate_schema()).unwrap();
        assert!(out["properties"].get("page_url").is_some());
    }

    #[test]
    fn defaults_apply() {
        let i = input(&["x"]);
        assert_eq!(i.language, "en");
        assert_eq!(i.max_results_per_query, 3);
        assert!(!i.full_text);
    }

    #[test]
    fn language_validation() {
        assert!(is_valid_language("en"));
        assert!(is_valid_language("zh-min-nan"));
        assert!(!is_valid_language(""));
        assert!(!is_valid_language("-en"));
        assert!(!is_valid_language("en-"));
        assert!(!is_valid_language("evil.com/x"));
        assert!(!is_valid_language(&"a".repeat(64)));
    }

    #[test]
    fn parse_titles_canned() {
        let data = json!({"query": {"search": [{"title": "A"}, {"title": "B"}]}});
        assert_eq!(WikipediaSearchTool::parse_titles(&data), vec!["A", "B"]);
        assert!(WikipediaSearchTool::parse_titles(&json!({})).is_empty());
    }

    #[test]
    fn article_fallback_url() {
        let tool = tool_for("https://{lang}.wikipedia.org".to_string());
        let a = tool.article_from_summary(&json!({"extract": "e"}), "q", "Foo Bar", "fr", None);
        assert_eq!(a.page_url, "https://fr.wikipedia.org/wiki/Foo_Bar");
        assert_eq!(a.title, "Foo Bar");
        assert_eq!(a.summary, "e");
    }

    #[test]
    fn invalid_language_is_error() {
        let tool = WikipediaSearchTool::new(WikipediaSearchConfig::default());
        let mut i = input(&["x"]);
        i.language = "evil.com/".into();
        assert!(tool.run(i).is_err());
    }

    async fn mock_search(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/w/api.php"))
            .and(query_param("list", "search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "query": {"search": [{"title": "Rust (programming language)"}, {"title": "Missing"}]}
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/rest_v1/page/summary/Rust%20(programming%20language)"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "title": "Rust (programming language)",
                "extract": "Rust is a language.",
                "description": "General-purpose language",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Rust_(programming_language)"}},
                "thumbnail": {"source": "https://img/x.png"}
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/rest_v1/page/summary/Missing"))
            .respond_with(ResponseTemplate::new(404))
            .mount(server)
            .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_parses_canned_responses() {
        let server = MockServer::start().await;
        mock_search(&server).await;
        let tool = tool_for(server.uri());
        let out = tool.run(input(&["rust"])).unwrap();
        assert_eq!(out.results.len(), 1);
        let a = &out.results[0];
        assert_eq!(a.description.as_deref(), Some("General-purpose language"));
        assert_eq!(a.thumbnail_url.as_deref(), Some("https://img/x.png"));
        assert!(a.full_text.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn full_text_is_fetched_for_top_result() {
        let server = MockServer::start().await;
        mock_search(&server).await;
        Mock::given(method("GET"))
            .and(path("/w/api.php"))
            .and(query_param("prop", "extracts"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "query": {"pages": {"1": {"extract": "Full article body."}}}
            })))
            .mount(&server)
            .await;
        let tool = tool_for(server.uri());
        let mut i = input(&["rust"]);
        i.full_text = true;
        let out = tool.run(i).unwrap();
        assert_eq!(out.results[0].full_text.as_deref(), Some("Full article body."));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn search_http_error_drops_query() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let tool = tool_for(server.uri());
        let out = tool.run(input(&["rust"])).unwrap();
        assert!(out.results.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires network access"]
    async fn live_search() {
        let tool = WikipediaSearchTool::new(WikipediaSearchConfig::default());
        let out = tool.run(input(&["Rust programming language"])).unwrap();
        assert!(!out.results.is_empty());
    }
}
