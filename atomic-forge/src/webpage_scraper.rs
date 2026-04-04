use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Scrape a webpage and extract its content as plain text.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct WebpageScraperInput {
    /// The URL of the webpage to scrape.
    pub url: String,
    /// Whether to include hyperlinks in the output.
    #[serde(default = "default_true")]
    pub include_links: bool,
}

fn default_true() -> bool {
    true
}

/// Metadata about a scraped webpage.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct WebpageMetadata {
    /// The page title.
    pub title: String,
    /// The content author, if found.
    pub author: Option<String>,
    /// The meta description, if found.
    pub description: Option<String>,
    /// The domain name.
    pub domain: String,
}

/// Output of the webpage scraper.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct WebpageScraperOutput {
    /// The extracted text content.
    pub content: String,
    /// Metadata about the page.
    pub metadata: WebpageMetadata,
    /// Error message if scraping partially failed.
    pub error: Option<String>,
}

/// Configuration for the webpage scraper.
pub struct WebpageScraperConfig {
    pub component: ComponentConfig,
    pub user_agent: String,
    pub timeout_secs: u64,
    pub max_content_length: usize,
}

impl Default for WebpageScraperConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("WebpageScraper")
                .with_description("Scrapes webpages and extracts text content"),
            user_agent: "Mozilla/5.0 (compatible; AtomicAgents/1.0)".to_string(),
            timeout_secs: 30,
            max_content_length: 1_000_000,
        }
    }
}

/// Tool for scraping webpages and extracting text content.
pub struct WebpageScraperTool {
    config: ComponentConfig,
    client: reqwest::Client,
    max_content_length: usize,
}

impl WebpageScraperTool {
    pub fn new(scraper_config: WebpageScraperConfig) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(&scraper_config.user_agent)
            .timeout(std::time::Duration::from_secs(scraper_config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config: scraper_config.component,
            client,
            max_content_length: scraper_config.max_content_length,
        }
    }

    async fn scrape(&self, url: &str, include_links: bool) -> std::result::Result<WebpageScraperOutput, Error> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("Failed to fetch URL: {e}")))?;

        let final_url = response.url().clone();
        let domain = final_url.host_str().unwrap_or("").to_string();

        let html = response
            .text()
            .await
            .map_err(|e| Error::Validation(format!("Failed to read response: {e}")))?;

        if html.len() > self.max_content_length {
            return Err(Error::Validation(format!(
                "Content exceeds maximum length of {} bytes",
                self.max_content_length
            )));
        }

        let document = scraper::Html::parse_document(&html);

        // Extract title
        let title_selector = scraper::Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>())
            .unwrap_or_default()
            .trim()
            .to_string();

        // Extract meta description
        let meta_selector = scraper::Selector::parse("meta[name=description]").unwrap();
        let description = document
            .select(&meta_selector)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(String::from);

        // Extract author
        let author_selector = scraper::Selector::parse("meta[name=author]").unwrap();
        let author = document
            .select(&author_selector)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(String::from);

        // Extract main text content
        let body_selector = scraper::Selector::parse("body").unwrap();
        let content = if let Some(body) = document.select(&body_selector).next() {
            extract_text(body, include_links)
        } else {
            String::new()
        };

        Ok(WebpageScraperOutput {
            content,
            metadata: WebpageMetadata {
                title,
                author,
                description,
                domain,
            },
            error: None,
        })
    }
}

impl Default for WebpageScraperTool {
    fn default() -> Self {
        Self::new(WebpageScraperConfig::default())
    }
}

impl Tool for WebpageScraperTool {
    type Input = WebpageScraperInput;
    type Output = WebpageScraperOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.scrape(&input.url, input.include_links))
        })
    }
}

/// Extracts text from an HTML element, optionally preserving links.
fn extract_text(element: scraper::ElementRef, include_links: bool) -> String {
    let mut text = String::new();
    // Remove script and style elements by skipping them
    let skip_tags = ["script", "style", "noscript", "nav", "footer", "header"];

    for node in element.children() {
        match node.value() {
            scraper::Node::Text(t) => {
                let trimmed = t.text.trim();
                if !trimmed.is_empty() {
                    text.push_str(trimmed);
                    text.push(' ');
                }
            }
            scraper::Node::Element(el) => {
                let tag = el.name();
                if skip_tags.contains(&tag) {
                    continue;
                }

                if let Some(child_ref) = scraper::ElementRef::wrap(node) {
                    let child_text = extract_text(child_ref, include_links);

                    if include_links && tag == "a" {
                        if let Some(href) = el.attr("href") {
                            if !child_text.trim().is_empty() {
                                text.push_str(&format!("[{}]({})", child_text.trim(), href));
                                text.push(' ');
                                continue;
                            }
                        }
                    }

                    text.push_str(&child_text);

                    // Add newlines after block elements
                    if matches!(
                        tag,
                        "p" | "div" | "br" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                            | "li" | "tr" | "blockquote"
                    ) {
                        text.push('\n');
                    }
                }
            }
            _ => {}
        }
    }

    // Clean up whitespace
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    lines.join("\n")
}
