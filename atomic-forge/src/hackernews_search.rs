use std::time::Duration;

use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const HN_ITEM_BASE: &str = "https://news.ycombinator.com/item?id=";

/// Type of Hacker News item to search for.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HackerNewsTag {
    /// Regular submissions.
    #[default]
    Story,
    /// HN comments.
    Comment,
    /// Show HN posts.
    ShowHn,
    /// Ask HN posts.
    AskHn,
    /// Polls.
    Poll,
    /// Stories that reached the front page.
    FrontPage,
}

impl HackerNewsTag {
    fn as_str(self) -> &'static str {
        match self {
            Self::Story => "story",
            Self::Comment => "comment",
            Self::ShowHn => "show_hn",
            Self::AskHn => "ask_hn",
            Self::Poll => "poll",
            Self::FrontPage => "front_page",
        }
    }
}

/// Result ordering for Hacker News searches.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HackerNewsSortBy {
    /// Algolia's relevance ranking.
    #[default]
    Relevance,
    /// Newest items first.
    Date,
}

fn default_max_results() -> u32 {
    10
}

/// Search Hacker News stories, comments, Show HN posts, Ask HN posts, and polls
/// via the free Algolia HN search API. No authentication required. Use this to
/// find tech news, discussions, and threads on a topic.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct HackerNewsSearchInput {
    /// Search queries to run against Hacker News.
    pub queries: Vec<String>,
    /// Type of item to search for: 'story', 'comment', 'show_hn', 'ask_hn',
    /// 'poll', or 'front_page'.
    #[serde(default)]
    pub tags: HackerNewsTag,
    /// 'relevance' uses Algolia's relevance ranking; 'date' returns the newest
    /// items first (uses the search_by_date endpoint).
    #[serde(default)]
    pub sort_by: HackerNewsSortBy,
    /// Maximum results per query (1-50).
    #[serde(default = "default_max_results")]
    pub max_results_per_query: u32,
    /// Optional Algolia numeric filter expression to narrow results, e.g.
    /// 'points>100', 'created_at_i>1700000000', or 'num_comments>50,points>20'.
    #[serde(default)]
    pub numeric_filters: Option<String>,
}

/// A single Hacker News item (story, comment, or other).
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct HackerNewsItem {
    /// The query that produced this item.
    pub query: String,
    /// Hacker News item id.
    pub object_id: String,
    /// Title (stories only; absent for comments).
    pub title: Option<String>,
    /// External URL submitted to HN (absent for Ask HN posts and comments).
    pub url: Option<String>,
    /// Direct URL to the item on news.ycombinator.com.
    pub hn_url: String,
    /// Username of the poster.
    pub author: Option<String>,
    /// Score of the item, if applicable.
    pub points: Option<i64>,
    /// Number of comments on the story, if applicable.
    pub num_comments: Option<i64>,
    /// ISO 8601 timestamp the item was posted.
    pub created_at: Option<String>,
    /// Body text for Ask HN posts and self-posts (HTML).
    pub story_text: Option<String>,
    /// Body text for comments (HTML).
    pub comment_text: Option<String>,
    /// Parent item id (for comments).
    pub parent_id: Option<String>,
    /// Story this comment belongs to (for comments).
    pub story_id: Option<String>,
}

/// A query that could not be completed, distinct from a successful empty search.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct HackerNewsSearchFailure {
    /// The query that failed.
    pub query: String,
    /// Reason the query could not be completed.
    pub error: String,
}

/// Output of the Hacker News search tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct HackerNewsSearchOutput {
    /// Matching items across all queries.
    pub results: Vec<HackerNewsItem>,
    /// Queries that failed; an empty results list is conclusive only when this is empty.
    #[serde(default)]
    pub failures: Vec<HackerNewsSearchFailure>,
}

/// Configuration for the Hacker News search tool.
pub struct HackerNewsSearchConfig {
    pub component: ComponentConfig,
    /// Algolia HN search base URL.
    pub base_url: String,
    /// User agent for HTTP requests.
    pub user_agent: String,
    /// HTTP request timeout.
    pub timeout: Duration,
}

impl Default for HackerNewsSearchConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("HackerNewsSearch")
                .with_description("Search Hacker News via the Algolia API"),
            base_url: "https://hn.algolia.com/api/v1".to_string(),
            user_agent:
                "atomic-agents-hackernews-tool/1.0 (+https://github.com/savvagent/atomic-agents-rs)"
                    .to_string(),
            timeout: Duration::from_secs(15),
        }
    }
}

/// Tool for searching Hacker News via the public Algolia API.
///
/// # Example
///
/// ```no_run
/// use atomic_agents::prelude::*;
/// use atomic_forge::hackernews_search::*;
///
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let tool = HackerNewsSearchTool::new(HackerNewsSearchConfig::default());
/// let out = tool.run(serde_json::from_str(r#"{"queries": ["rust"]}"#).unwrap()).unwrap();
/// println!("{}", out.results.len());
/// # }
/// ```
pub struct HackerNewsSearchTool {
    config: ComponentConfig,
    base_url: String,
    client: reqwest::Client,
}

fn first_str(hit: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|k| hit.get(*k).and_then(Value::as_str))
        .find(|s| !s.is_empty())
        .map(String::from)
}

/// Render a JSON string or number as a string (Algolia returns ids as either).
fn id_string(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

impl HackerNewsSearchTool {
    pub fn new(config: HackerNewsSearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(config.user_agent)
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self {
            config: config.component,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            client,
        }
    }

    /// Convert one Algolia hit into a [`HackerNewsItem`].
    pub fn to_item(hit: &Value, query: &str) -> HackerNewsItem {
        let object_id = id_string(hit.get("objectID"))
            .filter(|s| !s.is_empty())
            .or_else(|| id_string(hit.get("story_id")))
            .unwrap_or_default();
        let hn_url = if object_id.is_empty() {
            "https://news.ycombinator.com/".to_string()
        } else {
            format!("{HN_ITEM_BASE}{object_id}")
        };
        HackerNewsItem {
            query: query.to_string(),
            object_id,
            title: first_str(hit, &["title", "story_title"]),
            url: first_str(hit, &["url", "story_url"]),
            hn_url,
            author: first_str(hit, &["author"]),
            points: hit.get("points").and_then(Value::as_i64),
            num_comments: hit.get("num_comments").and_then(Value::as_i64),
            created_at: first_str(hit, &["created_at"]),
            story_text: first_str(hit, &["story_text"]),
            comment_text: first_str(hit, &["comment_text"]),
            parent_id: id_string(hit.get("parent_id")),
            story_id: id_string(hit.get("story_id")),
        }
    }

    async fn fetch(
        &self,
        query: &str,
        input: &HackerNewsSearchInput,
    ) -> std::result::Result<Vec<HackerNewsItem>, String> {
        let endpoint = match input.sort_by {
            HackerNewsSortBy::Date => "search_by_date",
            HackerNewsSortBy::Relevance => "search",
        };
        let mut params = vec![
            ("query", query.to_string()),
            ("tags", input.tags.as_str().to_string()),
            (
                "hitsPerPage",
                input.max_results_per_query.clamp(1, 50).to_string(),
            ),
        ];
        if let Some(nf) = input.numeric_filters.as_deref().filter(|s| !s.is_empty()) {
            params.push(("numericFilters", nf.to_string()));
        }
        let response = self
            .client
            .get(format!("{}/{endpoint}", self.base_url))
            .header("Accept", "application/json")
            .query(&params)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("HN search failed for '{query}': {status}"));
        }
        let data: Value = response
            .json()
            .await
            .map_err(|e| format!("invalid JSON response: {e}"))?;
        let hits = data
            .get("hits")
            .and_then(Value::as_array)
            .ok_or_else(|| "HN search returned an invalid hits response".to_string())?;
        Ok(hits.iter().map(|h| Self::to_item(h, query)).collect())
    }
}

impl Tool for HackerNewsSearchTool {
    type Input = HackerNewsSearchInput;
    type Output = HackerNewsSearchOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let (results, failures) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let groups =
                    futures::future::join_all(input.queries.iter().map(|q| self.fetch(q, &input)))
                        .await;
                let mut results = Vec::new();
                let mut failures = Vec::new();
                for (query, group) in input.queries.iter().zip(groups) {
                    match group {
                        Ok(items) => results.extend(items),
                        Err(error) => {
                            tracing::warn!("HN query '{query}' failed: {error}");
                            failures.push(HackerNewsSearchFailure {
                                query: query.clone(),
                                error,
                            });
                        }
                    }
                }
                (results, failures)
            })
        });
        Ok(HackerNewsSearchOutput { results, failures })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn input(queries: &[&str]) -> HackerNewsSearchInput {
        serde_json::from_value(json!({ "queries": queries })).unwrap()
    }

    fn tool_for(url: String) -> HackerNewsSearchTool {
        HackerNewsSearchTool::new(HackerNewsSearchConfig {
            base_url: url,
            ..Default::default()
        })
    }

    #[test]
    fn schema_metadata() {
        assert!(HackerNewsSearchInput::schema_description().contains("Hacker News"));
        let schema = serde_json::to_value(HackerNewsSearchInput::generate_schema()).unwrap();
        assert!(schema["properties"].get("numeric_filters").is_some());
        let out = serde_json::to_value(HackerNewsSearchOutput::generate_schema()).unwrap();
        assert!(out["properties"].get("failures").is_some());
    }

    #[test]
    fn input_defaults_and_enum_parsing() {
        let i = input(&["a"]);
        assert_eq!(i.tags, HackerNewsTag::Story);
        assert_eq!(i.sort_by, HackerNewsSortBy::Relevance);
        assert_eq!(i.max_results_per_query, 10);
        let j: HackerNewsSearchInput =
            serde_json::from_value(json!({"queries": ["a"], "tags": "show_hn", "sort_by": "date"}))
                .unwrap();
        assert_eq!(j.tags, HackerNewsTag::ShowHn);
        assert!(
            serde_json::from_value::<HackerNewsSearchInput>(
                json!({"queries": ["a"], "tags": "nope"})
            )
            .is_err()
        );
    }

    #[test]
    fn to_item_story_and_comment() {
        let story = json!({
            "objectID": "123", "title": "Hello", "url": "https://x.dev", "author": "pg",
            "points": 42, "num_comments": 7, "created_at": "2024-01-01T00:00:00Z"
        });
        let item = HackerNewsSearchTool::to_item(&story, "q");
        assert_eq!(item.hn_url, "https://news.ycombinator.com/item?id=123");
        assert_eq!(item.title.as_deref(), Some("Hello"));
        assert_eq!(item.points, Some(42));

        let comment = json!({
            "objectID": "9", "comment_text": "<p>hi</p>", "story_title": "Parent",
            "story_url": "https://p.dev", "parent_id": 8, "story_id": 7
        });
        let item = HackerNewsSearchTool::to_item(&comment, "q");
        assert_eq!(item.title.as_deref(), Some("Parent"));
        assert_eq!(item.url.as_deref(), Some("https://p.dev"));
        assert_eq!(item.parent_id.as_deref(), Some("8"));
        assert_eq!(item.story_id.as_deref(), Some("7"));

        let empty = HackerNewsSearchTool::to_item(&json!({}), "q");
        assert_eq!(empty.hn_url, "https://news.ycombinator.com/");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_parses_canned_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search_by_date"))
            .and(query_param("tags", "story"))
            .and(query_param("numericFilters", "points>10"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "hits": [{"objectID": "1", "title": "T", "points": 11}]
            })))
            .mount(&server)
            .await;
        let tool = tool_for(server.uri());
        let mut i = input(&["rust"]);
        i.sort_by = HackerNewsSortBy::Date;
        i.numeric_filters = Some("points>10".into());
        let out = tool.run(i).unwrap();
        assert_eq!(out.results.len(), 1);
        assert!(out.failures.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failures_are_reported() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(query_param("query", "bad"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(query_param("query", "worse"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"nohits": true})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(query_param("query", "good"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"hits": []})))
            .mount(&server)
            .await;
        let tool = tool_for(server.uri());
        let out = tool.run(input(&["bad", "worse", "good"])).unwrap();
        assert!(out.results.is_empty());
        assert_eq!(out.failures.len(), 2);
        assert_eq!(out.failures[0].query, "bad");
        assert!(out.failures[0].error.contains("500"));
        assert!(out.failures[1].error.contains("invalid hits"));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires network access"]
    async fn live_search() {
        let tool = HackerNewsSearchTool::new(HackerNewsSearchConfig::default());
        let out = tool.run(input(&["rust"])).unwrap();
        assert!(!out.results.is_empty());
    }
}
