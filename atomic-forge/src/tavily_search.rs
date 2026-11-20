use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Search for information using Tavily. Returns a list of search results
/// with descriptions and URLs for further exploration.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct TavilySearchInput {
    /// List of search queries.
    pub queries: Vec<String>,
}

/// A single search result item from Tavily.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct TavilySearchResultItem {
    /// The title of the search result.
    pub title: String,
    /// The URL of the search result.
    pub url: String,
    /// The content snippet of the search result.
    pub content: Option<String>,
    /// The relevance score of the search result.
    pub score: f64,
    /// The query used to obtain this result.
    pub query: Option<String>,
    /// Direct answer provided by Tavily.
    pub answer: Option<String>,
}

/// Output of the Tavily search tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct TavilySearchOutput {
    /// List of search result items.
    pub results: Vec<TavilySearchResultItem>,
}

/// Configuration for the Tavily search tool.
pub struct TavilySearchConfig {
    pub component: ComponentConfig,
    pub api_key: String,
    pub max_results: usize,
    pub search_depth: String,
    pub include_domains: Option<Vec<String>>,
    pub exclude_domains: Option<Vec<String>>,
}

impl TavilySearchConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("TavilySearch")
                .with_description("Search the web using Tavily"),
            api_key: api_key.into(),
            max_results: 5,
            search_depth: "basic".to_string(),
            include_domains: None,
            exclude_domains: None,
        }
    }
}

/// Tool for searching the web using the Tavily API.
pub struct TavilySearchTool {
    config: ComponentConfig,
    api_key: String,
    max_results: usize,
    search_depth: String,
    include_domains: Option<Vec<String>>,
    exclude_domains: Option<Vec<String>>,
    client: reqwest::Client,
}

impl TavilySearchTool {
    pub fn new(config: TavilySearchConfig) -> Self {
        let api_key = if config.api_key.is_empty() {
            std::env::var("TAVILY_API_KEY").unwrap_or_default()
        } else {
            config.api_key
        };

        Self {
            config: config.component,
            api_key,
            max_results: config.max_results,
            search_depth: config.search_depth,
            include_domains: config.include_domains,
            exclude_domains: config.exclude_domains,
            client: reqwest::Client::new(),
        }
    }

    async fn search_query(
        &self,
        query: &str,
    ) -> std::result::Result<Vec<TavilySearchResultItem>, Error> {
        let mut body = serde_json::json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
        });

        if let Some(domains) = &self.include_domains {
            body["include_domains"] = serde_json::json!(domains);
        }
        if let Some(domains) = &self.exclude_domains {
            body["exclude_domains"] = serde_json::json!(domains);
        }

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("Tavily API request failed: {e}")))?;

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse Tavily response: {e}")))?;

        let answer = result
            .get("answer")
            .and_then(|a| a.as_str())
            .map(String::from);

        let items = result
            .get("results")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|item| TavilySearchResultItem {
                        title: item
                            .get("title")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string(),
                        url: item
                            .get("url")
                            .and_then(|u| u.as_str())
                            .unwrap_or("")
                            .to_string(),
                        content: item
                            .get("content")
                            .and_then(|c| c.as_str())
                            .map(String::from),
                        score: item.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0),
                        query: Some(query.to_string()),
                        answer: answer.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(items)
    }
}

impl Tool for TavilySearchTool {
    type Input = TavilySearchInput;
    type Output = TavilySearchOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut all_results = Vec::new();
                for query in &input.queries {
                    match self.search_query(query).await {
                        Ok(items) => all_results.extend(items),
                        Err(e) => tracing::warn!("Search query failed: {e}"),
                    }
                }
                all_results
            })
        });

        Ok(TavilySearchOutput { results })
    }
}
