use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Search for information using the BoCha API.
/// Specialized for multilingual web searches.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct BochaSearchInput {
    /// List of search queries.
    pub queries: Vec<String>,
}

/// A single search result from BoCha.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct BochaSearchResultItem {
    /// The title of the search result.
    pub name: String,
    /// The URL of the search result.
    pub url: String,
    /// Content snippet.
    pub snippet: Option<String>,
    /// The query that produced this result.
    pub query: Option<String>,
}

/// Output of the BoCha search tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct BochaSearchOutput {
    /// List of search result items.
    pub results: Vec<BochaSearchResultItem>,
}

/// Configuration for the BoCha search tool.
pub struct BochaSearchConfig {
    pub component: ComponentConfig,
    pub api_key: String,
    pub freshness: String,
    pub count: usize,
    pub include: Option<String>,
    pub exclude: Option<String>,
}

impl BochaSearchConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("BoChaSearch")
                .with_description("Web search via BoCha API"),
            api_key: api_key.into(),
            freshness: "noLimit".to_string(),
            count: 10,
            include: None,
            exclude: None,
        }
    }
}

/// Tool for web search using the BoCha API.
pub struct BochaSearchTool {
    config: ComponentConfig,
    api_key: String,
    freshness: String,
    count: usize,
    include: Option<String>,
    exclude: Option<String>,
    client: reqwest::Client,
}

impl BochaSearchTool {
    pub fn new(search_config: BochaSearchConfig) -> Self {
        let api_key = if search_config.api_key.is_empty() {
            std::env::var("BOCHA_API_KEY").unwrap_or_default()
        } else {
            search_config.api_key
        };

        Self {
            config: search_config.component,
            api_key,
            freshness: search_config.freshness,
            count: search_config.count,
            include: search_config.include,
            exclude: search_config.exclude,
            client: reqwest::Client::new(),
        }
    }

    async fn search_query(&self, query: &str) -> std::result::Result<Vec<BochaSearchResultItem>, Error> {
        let mut body = serde_json::json!({
            "q": query,
            "freshness": self.freshness,
            "count": self.count,
        });

        if let Some(include) = &self.include {
            body["include"] = serde_json::json!(include);
        }
        if let Some(exclude) = &self.exclude {
            body["exclude"] = serde_json::json!(exclude);
        }

        let response = self
            .client
            .post("https://api.bochaai.com/v1/web/search")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("BoCha API request failed: {e}")))?;

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse BoCha response: {e}")))?;

        let items = result
            .get("web")
            .and_then(|w| w.get("results"))
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|item| BochaSearchResultItem {
                        name: item.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string(),
                        url: item.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string(),
                        snippet: item.get("snippet").and_then(|s| s.as_str()).map(String::from),
                        query: Some(query.to_string()),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(items)
    }
}

impl Tool for BochaSearchTool {
    type Input = BochaSearchInput;
    type Output = BochaSearchOutput;

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
                        Err(e) => tracing::warn!("BoCha query failed: {e}"),
                    }
                }
                all_results
            })
        });

        Ok(BochaSearchOutput { results })
    }
}
