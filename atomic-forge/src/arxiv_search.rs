use std::time::Duration;

use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const ATOM_NS: &str = "http://www.w3.org/2005/Atom";
const ARXIV_NS: &str = "http://arxiv.org/schemas/atom";

/// Field prefixes that mark a query as an explicit arXiv field expression.
const FIELD_PREFIXES: [&str; 10] = [
    "ti:",
    "au:",
    "abs:",
    "co:",
    "jr:",
    "cat:",
    "rn:",
    "id:",
    "all:",
    "submittedDate:",
];

/// Sort key for arXiv results.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ArxivSortBy {
    /// Rank by relevance.
    #[default]
    Relevance,
    /// Sort by last update date.
    LastUpdatedDate,
    /// Sort by submission date.
    SubmittedDate,
}

impl ArxivSortBy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Relevance => "relevance",
            Self::LastUpdatedDate => "lastUpdatedDate",
            Self::SubmittedDate => "submittedDate",
        }
    }
}

/// Sort direction for arXiv results.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ArxivSortOrder {
    /// Smallest first.
    Ascending,
    /// Largest first.
    #[default]
    Descending,
}

impl ArxivSortOrder {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ascending => "ascending",
            Self::Descending => "descending",
        }
    }
}

fn default_max_results() -> u32 {
    5
}

/// Search arXiv for academic papers. Each query is sent to the public arXiv API
/// (no key needed) and the matching papers are returned with title, authors,
/// abstract, dates, categories, and links to the abstract page and PDF.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct ArxivSearchInput {
    /// Search queries to run. Use plain keywords (e.g. 'retrieval augmented
    /// generation') or arXiv field-prefixed expressions (e.g. 'ti:LLM AND abs:agents').
    pub queries: Vec<String>,
    /// Maximum number of papers per query (1-50).
    #[serde(default = "default_max_results")]
    pub max_results_per_query: u32,
    /// Sort order key for results.
    #[serde(default)]
    pub sort_by: ArxivSortBy,
    /// Direction for the chosen sort.
    #[serde(default)]
    pub sort_order: ArxivSortOrder,
    /// Optional arXiv category to filter to (e.g. 'cs.AI', 'cs.CL', 'stat.ML').
    /// Combined with the query as `cat:<value>`.
    #[serde(default)]
    pub category: Option<String>,
}

/// A single paper returned by arXiv.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct ArxivPaper {
    /// The query that produced this paper.
    pub query: String,
    /// arXiv identifier (e.g. '2401.12345').
    pub arxiv_id: String,
    /// Paper title.
    pub title: String,
    /// Abstract / summary.
    pub summary: String,
    /// Author names.
    pub authors: Vec<String>,
    /// First publication timestamp (ISO 8601).
    pub published: String,
    /// Most recent update timestamp (ISO 8601).
    pub updated: String,
    /// arXiv categories (e.g. ['cs.AI']).
    pub categories: Vec<String>,
    /// Primary arXiv category.
    pub primary_category: Option<String>,
    /// Direct URL to the PDF.
    pub pdf_url: String,
    /// URL to the abstract page.
    pub abs_url: String,
    /// Author-supplied comment, if any.
    pub comment: Option<String>,
    /// Journal reference, if provided.
    pub journal_ref: Option<String>,
}

/// Output of the arXiv search tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct ArxivSearchOutput {
    /// Matching papers across all queries.
    pub results: Vec<ArxivPaper>,
}

/// Configuration for the arXiv search tool.
pub struct ArxivSearchConfig {
    pub component: ComponentConfig,
    /// Base URL of the arXiv API.
    pub base_url: String,
    /// User agent string for arXiv requests.
    pub user_agent: String,
    /// HTTP request timeout.
    pub timeout: Duration,
}

impl Default for ArxivSearchConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("ArxivSearch")
                .with_description("Search arXiv for academic papers"),
            base_url: "https://export.arxiv.org/api/query".to_string(),
            user_agent:
                "atomic-agents-arxiv-tool/1.0 (+https://github.com/savvagent/atomic-agents-rs)"
                    .to_string(),
            timeout: Duration::from_secs(20),
        }
    }
}

/// Tool for searching arXiv (free public API).
///
/// # Example
///
/// ```no_run
/// use atomic_agents::prelude::*;
/// use atomic_forge::arxiv_search::*;
///
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let tool = ArxivSearchTool::new(ArxivSearchConfig::default());
/// let out = tool.run(ArxivSearchInput {
///     queries: vec!["retrieval augmented generation".into()],
///     max_results_per_query: 2,
///     sort_by: Default::default(),
///     sort_order: Default::default(),
///     category: Some("cs.AI".into()),
/// }).unwrap();
/// println!("{}", out.results.len());
/// # }
/// ```
pub struct ArxivSearchTool {
    config: ComponentConfig,
    base_url: String,
    client: reqwest::Client,
}

fn is_atom<'a, 'i>(node: &roxmltree::Node<'a, 'i>, ns: &str, name: &str) -> bool {
    node.is_element() && node.tag_name().name() == name && node.tag_name().namespace() == Some(ns)
}

fn children_named<'a, 'i: 'a>(
    node: roxmltree::Node<'a, 'i>,
    ns: &'a str,
    name: &'a str,
) -> impl Iterator<Item = roxmltree::Node<'a, 'i>> + 'a {
    node.children().filter(move |c| is_atom(c, ns, name))
}

fn child_text(node: roxmltree::Node<'_, '_>, ns: &str, name: &str) -> Option<String> {
    children_named(node, ns, name)
        .next()
        .map(|c| c.text().unwrap_or("").trim().to_string())
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl ArxivSearchTool {
    pub fn new(config: ArxivSearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(config.user_agent)
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self {
            config: config.component,
            base_url: config.base_url,
            client,
        }
    }

    /// Build the arXiv `search_query` value for a user query and optional category.
    ///
    /// ```
    /// use atomic_forge::arxiv_search::ArxivSearchTool;
    /// assert_eq!(ArxivSearchTool::build_search_query("llm", None), "all:llm");
    /// assert_eq!(
    ///     ArxivSearchTool::build_search_query("ti:llm", Some("cs.AI")),
    ///     "(ti:llm) AND cat:cs.AI"
    /// );
    /// ```
    pub fn build_search_query(query: &str, category: Option<&str>) -> String {
        let base = if FIELD_PREFIXES.iter().any(|p| query.contains(p)) {
            query.to_string()
        } else {
            format!("all:{query}")
        };
        match category.filter(|c| !c.is_empty()) {
            // Group the query: arXiv binds `all:` to the first word only and gives AND precedence.
            Some(cat) => format!("({base}) AND cat:{cat}"),
            None => base,
        }
    }

    /// Strip the URL prefix and version suffix from an arXiv entry id.
    ///
    /// ```
    /// use atomic_forge::arxiv_search::ArxivSearchTool;
    /// assert_eq!(ArxivSearchTool::extract_id("http://arxiv.org/abs/2401.12345v2"), "2401.12345");
    /// ```
    pub fn extract_id(entry_id: &str) -> String {
        let tail = entry_id.rsplit("/abs/").next().unwrap_or(entry_id);
        if let Some(pos) = tail.rfind('v') {
            let (head, ver) = tail.split_at(pos);
            if ver.len() > 1 && ver[1..].bytes().all(|b| b.is_ascii_digit()) {
                return head.to_string();
            }
        }
        tail.to_string()
    }

    /// Parse an Atom feed returned by arXiv into papers.
    pub fn parse_feed(xml: &str, query: &str) -> std::result::Result<Vec<ArxivPaper>, Error> {
        let doc = roxmltree::Document::parse(xml)
            .map_err(|e| Error::Validation(format!("Failed to parse arXiv feed: {e}")))?;
        Ok(children_named(doc.root_element(), ATOM_NS, "entry")
            .map(|entry| Self::parse_entry(entry, query))
            .collect())
    }

    fn parse_entry(entry: roxmltree::Node<'_, '_>, query: &str) -> ArxivPaper {
        let entry_id = child_text(entry, ATOM_NS, "id").unwrap_or_default();
        let authors = children_named(entry, ATOM_NS, "author")
            .filter_map(|a| child_text(a, ATOM_NS, "name"))
            .filter(|n| !n.is_empty())
            .collect();
        let categories = children_named(entry, ATOM_NS, "category")
            .filter_map(|c| c.attribute("term"))
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect();
        let primary_category = children_named(entry, ARXIV_NS, "primary_category")
            .next()
            .and_then(|n| n.attribute("term"))
            .map(String::from);

        let mut pdf_url = String::new();
        let mut abs_url = entry_id.clone();
        for link in children_named(entry, ATOM_NS, "link") {
            if link.attribute("title") == Some("pdf")
                || link.attribute("type") == Some("application/pdf")
            {
                pdf_url = link.attribute("href").unwrap_or("").to_string();
            } else if link.attribute("rel") == Some("alternate")
                && let Some(h) = link.attribute("href")
            {
                abs_url = h.to_string();
            }
        }
        if pdf_url.is_empty() && !abs_url.is_empty() {
            pdf_url = abs_url.replace("/abs/", "/pdf/");
        }

        let non_empty = |s: Option<String>| s.filter(|s| !s.is_empty());

        ArxivPaper {
            query: query.to_string(),
            arxiv_id: Self::extract_id(&entry_id),
            title: collapse_ws(&child_text(entry, ATOM_NS, "title").unwrap_or_default()),
            summary: collapse_ws(&child_text(entry, ATOM_NS, "summary").unwrap_or_default()),
            authors,
            published: child_text(entry, ATOM_NS, "published").unwrap_or_default(),
            updated: child_text(entry, ATOM_NS, "updated").unwrap_or_default(),
            categories,
            primary_category,
            pdf_url,
            abs_url,
            comment: non_empty(child_text(entry, ARXIV_NS, "comment")),
            journal_ref: non_empty(child_text(entry, ARXIV_NS, "journal_ref")),
        }
    }

    async fn fetch(
        &self,
        query: &str,
        input: &ArxivSearchInput,
    ) -> std::result::Result<Vec<ArxivPaper>, Error> {
        let max_results = input.max_results_per_query.clamp(1, 50);
        let params = [
            (
                "search_query",
                Self::build_search_query(query, input.category.as_deref()),
            ),
            ("start", "0".to_string()),
            ("max_results", max_results.to_string()),
            ("sortBy", input.sort_by.as_str().to_string()),
            ("sortOrder", input.sort_order.as_str().to_string()),
        ];
        let response = self
            .client
            .get(&self.base_url)
            .header("Accept", "application/atom+xml")
            .query(&params)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("arXiv request failed: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Validation(format!(
                "arXiv search failed for '{query}': {status}"
            )));
        }
        let text = response
            .text()
            .await
            .map_err(|e| Error::Validation(format!("Failed to read arXiv response: {e}")))?;
        Self::parse_feed(&text, query)
    }
}

impl Tool for ArxivSearchTool {
    type Input = ArxivSearchInput;
    type Output = ArxivSearchOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let groups =
                    futures::future::join_all(input.queries.iter().map(|q| self.fetch(q, &input)))
                        .await;
                let mut all = Vec::new();
                for (query, group) in input.queries.iter().zip(groups) {
                    match group {
                        Ok(papers) => all.extend(papers),
                        Err(e) => tracing::warn!("arXiv query '{query}' failed: {e}"),
                    }
                }
                all
            })
        });
        Ok(ArxivSearchOutput { results })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const FEED: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-02-01T00:00:00Z</updated>
    <published>2024-01-22T00:00:00Z</published>
    <title>Agents
      for Everyone</title>
    <summary>  An abstract
      spanning lines. </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:comment>Accepted at TestConf</arxiv:comment>
    <arxiv:journal_ref>J. Test 1 (2024)</arxiv:journal_ref>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"#;

    fn input(queries: Vec<&str>) -> ArxivSearchInput {
        ArxivSearchInput {
            queries: queries.into_iter().map(String::from).collect(),
            max_results_per_query: 5,
            sort_by: ArxivSortBy::default(),
            sort_order: ArxivSortOrder::default(),
            category: None,
        }
    }

    fn tool_for(url: String) -> ArxivSearchTool {
        let cfg = ArxivSearchConfig {
            base_url: url,
            ..Default::default()
        };
        ArxivSearchTool::new(cfg)
    }

    #[test]
    fn schema_metadata() {
        assert!(ArxivSearchInput::schema_description().contains("arXiv"));
        let schema = serde_json::to_value(ArxivSearchInput::generate_schema()).unwrap();
        assert!(schema["properties"].get("queries").is_some());
        assert!(schema["properties"].get("sort_by").is_some());
        let out = serde_json::to_value(ArxivSearchOutput::generate_schema()).unwrap();
        assert!(out["properties"].get("results").is_some());
    }

    #[test]
    fn input_defaults_apply() {
        let i: ArxivSearchInput = serde_json::from_str(r#"{"queries":["a"]}"#).unwrap();
        assert_eq!(i.max_results_per_query, 5);
        assert_eq!(i.sort_by, ArxivSortBy::Relevance);
        assert_eq!(i.sort_order, ArxivSortOrder::Descending);
        let j: ArxivSearchInput =
            serde_json::from_str(r#"{"queries":["a"],"sort_by":"lastUpdatedDate"}"#).unwrap();
        assert_eq!(j.sort_by, ArxivSortBy::LastUpdatedDate);
        assert!(
            serde_json::from_str::<ArxivSearchInput>(r#"{"queries":["a"],"sort_by":"bogus"}"#)
                .is_err()
        );
    }

    #[test]
    fn build_query_variants() {
        assert_eq!(
            ArxivSearchTool::build_search_query("diffusion model", None),
            "all:diffusion model"
        );
        assert_eq!(
            ArxivSearchTool::build_search_query("diffusion model", Some("cs.CV")),
            "(all:diffusion model) AND cat:cs.CV"
        );
        assert_eq!(
            ArxivSearchTool::build_search_query("au:smith", None),
            "au:smith"
        );
    }

    #[test]
    fn extract_id_strips_version() {
        assert_eq!(
            ArxivSearchTool::extract_id("http://arxiv.org/abs/2401.12345v12"),
            "2401.12345"
        );
        assert_eq!(
            ArxivSearchTool::extract_id("http://arxiv.org/abs/hep-th/9901001v1"),
            "hep-th/9901001"
        );
        assert_eq!(
            ArxivSearchTool::extract_id("http://arxiv.org/abs/2401.12345"),
            "2401.12345"
        );
    }

    #[test]
    fn parse_feed_canned() {
        let papers = ArxivSearchTool::parse_feed(FEED, "q").unwrap();
        assert_eq!(papers.len(), 1);
        let p = &papers[0];
        assert_eq!(p.arxiv_id, "2401.12345");
        assert_eq!(p.title, "Agents for Everyone");
        assert_eq!(p.summary, "An abstract spanning lines.");
        assert_eq!(p.authors, vec!["Ada Lovelace", "Alan Turing"]);
        assert_eq!(p.categories, vec!["cs.AI", "cs.CL"]);
        assert_eq!(p.primary_category.as_deref(), Some("cs.AI"));
        assert_eq!(p.pdf_url, "http://arxiv.org/pdf/2401.12345v2");
        assert_eq!(p.abs_url, "http://arxiv.org/abs/2401.12345v2");
        assert_eq!(p.comment.as_deref(), Some("Accepted at TestConf"));
        assert_eq!(p.journal_ref.as_deref(), Some("J. Test 1 (2024)"));
    }

    #[test]
    fn parse_feed_invalid_xml_is_error() {
        assert!(ArxivSearchTool::parse_feed("<feed", "q").is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_against_mock_server() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(query_param("search_query", "(all:rag) AND cat:cs.AI"))
            .and(query_param("sortBy", "submittedDate"))
            .respond_with(ResponseTemplate::new(200).set_body_string(FEED))
            .mount(&server)
            .await;
        let tool = tool_for(format!("{}/api/query", server.uri()));
        let mut i = input(vec!["rag"]);
        i.category = Some("cs.AI".into());
        i.sort_by = ArxivSortBy::SubmittedDate;
        let out = tool.run(i).unwrap();
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.results[0].query, "rag");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn http_error_drops_query() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let tool = tool_for(server.uri());
        let out = tool.run(input(vec!["x"])).unwrap();
        assert!(out.results.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires network access"]
    async fn live_search() {
        let tool = ArxivSearchTool::new(ArxivSearchConfig::default());
        let out = tool
            .run(input(vec!["retrieval augmented generation"]))
            .unwrap();
        assert!(!out.results.is_empty());
    }
}
