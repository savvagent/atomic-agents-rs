use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Search for information using SearXNG meta-search engine.
/// Returns results aggregated from multiple search engines.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct SearxngSearchInput {
    /// List of search queries.
    pub queries: Vec<String>,
    /// Category of the search (general, news, or social_media).
    pub category: Option<String>,
}

/// A single search result from SearXNG.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct SearxngSearchResultItem {
    /// The URL of the search result.
    pub url: String,
    /// The title of the search result.
    pub title: String,
    /// The content snippet.
    pub content: Option<String>,
    /// The query that produced this result.
    pub query: String,
}

/// Output of the SearXNG search tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct SearxngSearchOutput {
    /// List of search result items.
    pub results: Vec<SearxngSearchResultItem>,
    /// The category used for the search.
    pub category: Option<String>,
}

/// Configuration for the SearXNG search tool.
pub struct SearxngSearchConfig {
    pub component: ComponentConfig,
    pub base_url: String,
    pub max_results: usize,
}

impl SearxngSearchConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("SearXNGSearch")
                .with_description("Meta-search using SearXNG"),
            base_url: base_url.into(),
            max_results: 10,
        }
    }
}

/// Tool for searching using a SearXNG instance.
pub struct SearxngSearchTool {
    config: ComponentConfig,
    base_url: String,
    max_results: usize,
    client: reqwest::Client,
}

impl SearxngSearchTool {
    pub fn new(search_config: SearxngSearchConfig) -> Self {
        Self {
            config: search_config.component,
            base_url: search_config.base_url,
            max_results: search_config.max_results,
            client: reqwest::Client::new(),
        }
    }

    /// Maps tool-level category names to SearXNG category names.
    fn searxng_category(category: &str) -> &str {
        if category == "social_media" { "social media" } else { category }
    }

    async fn search_query(
        &self,
        query: &str,
        category: Option<&str>,
    ) -> std::result::Result<Vec<SearxngSearchResultItem>, Error> {
        let mut params = vec![
            ("q", query.to_string()),
            ("format", "json".to_string()),
        ];

        if let Some(cat) = category {
            params.push(("categories", Self::searxng_category(cat).to_string()));
        }

        let url = format!("{}/search", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("SearXNG request failed: {e}")))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse SearXNG response: {e}")))?;

        let results = body
            .get("results")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .take(self.max_results)
                    .map(|item| SearxngSearchResultItem {
                        url: item.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string(),
                        title: item.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string(),
                        content: item.get("content").and_then(|c| c.as_str()).map(String::from),
                        query: query.to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }
}

impl Tool for SearxngSearchTool {
    type Input = SearxngSearchInput;
    type Output = SearxngSearchOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let category = input.category.clone();
        let results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut all_results = Vec::new();
                for query in &input.queries {
                    match self.search_query(query, category.as_deref()).await {
                        Ok(items) => all_results.extend(items),
                        Err(e) => tracing::warn!("SearXNG query failed: {e}"),
                    }
                }
                all_results
            })
        });

        Ok(SearxngSearchOutput {
            results,
            category: input.category,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn social_media_maps_to_searxng_category() {
        assert_eq!(SearxngSearchTool::searxng_category("social_media"), "social media");
        assert_eq!(SearxngSearchTool::searxng_category("news"), "news");
        assert_eq!(SearxngSearchTool::searxng_category("general"), "general");
    }
}
