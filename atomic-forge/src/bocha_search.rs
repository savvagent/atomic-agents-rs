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

/// Time-filter presets accepted by the BoCha API.
const FRESHNESS_PRESETS: [&str; 5] = ["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"];

fn parse_iso_date(s: &str) -> Option<(u32, u32, u32)> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let digits = |r: std::ops::Range<usize>| -> Option<u32> {
        let part = &s[r];
        if part.bytes().all(|c| c.is_ascii_digit()) { part.parse().ok() } else { None }
    };
    let (y, m, d) = (digits(0..4)?, digits(5..7)?, digits(8..10)?);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let max_day = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if leap { 29 } else { 28 },
        _ => return None,
    };
    if y == 0 || d == 0 || d > max_day { None } else { Some((y, m, d)) }
}

/// Validates a BoCha `freshness` value: a preset, a real `YYYY-MM-DD` date, or
/// an ordered `YYYY-MM-DD..YYYY-MM-DD` range.
///
/// ```
/// use atomic_forge::bocha_search::validate_freshness;
/// assert!(validate_freshness("oneWeek").is_ok());
/// assert!(validate_freshness("2024-02-29..2024-03-01").is_ok());
/// assert!(validate_freshness("2023-02-29").is_err());
/// ```
pub fn validate_freshness(value: &str) -> std::result::Result<(), Error> {
    if FRESHNESS_PRESETS.contains(&value) {
        return Ok(());
    }
    let invalid = || {
        Error::Validation(
            "freshness must be a supported preset, a YYYY-MM-DD date, or an ordered YYYY-MM-DD..YYYY-MM-DD range"
                .to_string(),
        )
    };
    let mut parts = value.split("..");
    let first = parts.next().and_then(parse_iso_date).ok_or_else(invalid)?;
    match (parts.next(), parts.next()) {
        (None, _) => Ok(()),
        (Some(end), None) => {
            let end = parse_iso_date(end).ok_or_else(invalid)?;
            if first > end { Err(invalid()) } else { Ok(()) }
        }
        _ => Err(invalid()),
    }
}

impl BochaSearchConfig {
    /// Sets the freshness filter, validating it first.
    pub fn with_freshness(mut self, freshness: impl Into<String>) -> std::result::Result<Self, Error> {
        let freshness = freshness.into();
        validate_freshness(&freshness)?;
        self.freshness = freshness;
        Ok(self)
    }

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

    fn request_body(query: &str, freshness: &str, count: usize) -> serde_json::Value {
        serde_json::json!({
            "query": query,
            "freshness": freshness,
            "count": count,
        })
    }

    /// Runs the tool, overriding the configured result count for this call.
    /// `None` or `Some(0)` falls back to the configured count.
    pub fn run_with_count(&self, input: BochaSearchInput, count: Option<usize>) -> Result<BochaSearchOutput> {
        validate_freshness(&self.freshness)?;
        let count = count.filter(|c| *c > 0).unwrap_or(self.count);
        let results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut all_results = Vec::new();
                for query in &input.queries {
                    match self.search_query(query, count).await {
                        Ok(items) => all_results.extend(items),
                        Err(e) => tracing::warn!("BoCha query failed: {e}"),
                    }
                }
                all_results
            })
        });

        Ok(BochaSearchOutput { results })
    }

    async fn search_query(
        &self,
        query: &str,
        count: usize,
    ) -> std::result::Result<Vec<BochaSearchResultItem>, Error> {
        let mut body = Self::request_body(query, &self.freshness, count);

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
        self.run_with_count(input, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freshness_accepts_presets_dates_and_ranges() {
        for v in ["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear", "2024-02-29", "2024-01-01..2024-01-01", "2023-01-01..2024-12-31"] {
            assert!(validate_freshness(v).is_ok(), "{v}");
        }
    }

    #[test]
    fn freshness_rejects_invalid_values() {
        for v in ["", "yesterday", "2023-02-29", "2024-13-01", "2024-00-10", "2024-1-01", "2024-01-02..2024-01-01", "2024-01-01..", "2024-01-01..2024-01-02..2024-01-03", "+024-01-01", "oneweek"] {
            assert!(validate_freshness(v).is_err(), "{v}");
        }
    }

    #[test]
    fn with_freshness_validates() {
        assert!(BochaSearchConfig::new("k").with_freshness("2024-01-01..2024-02-01").is_ok());
        assert!(BochaSearchConfig::new("k").with_freshness("bad").is_err());
    }

    #[test]
    fn run_rejects_invalid_freshness_set_directly() {
        let mut cfg = BochaSearchConfig::new("k");
        cfg.freshness = "bad".into();
        let tool = BochaSearchTool::new(cfg);
        let out = tool.run(BochaSearchInput { queries: vec!["q".into()] });
        assert!(out.is_err());
    }

    #[test]
    fn request_body_uses_query_and_count() {
        let body = BochaSearchTool::request_body("rust", "oneDay", 3);
        assert_eq!(body["query"], "rust");
        assert_eq!(body["count"], 3);
        assert_eq!(body["freshness"], "oneDay");
    }
}
