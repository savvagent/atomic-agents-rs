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
        let mut response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("Failed to fetch URL: {e}")))?;

        let final_url = response.url().clone();
        let domain = final_url.host_str().unwrap_or("").to_string();

        // Stream the body so oversized responses are rejected without buffering them fully.
        if response.content_length().is_some_and(|l| l > self.max_content_length as u64) {
            return Err(too_large(self.max_content_length));
        }
        let mut body: Vec<u8> = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| Error::Validation(format!("Failed to read response: {e}")))?
        {
            if body.len() + chunk.len() > self.max_content_length {
                return Err(too_large(self.max_content_length));
            }
            body.extend_from_slice(&chunk);
        }
        let html = String::from_utf8_lossy(&body).into_owned();

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
        let base_selector = scraper::Selector::parse("base[href]").unwrap();
        let base_url = document
            .select(&base_selector)
            .next()
            .and_then(|el| el.value().attr("href"))
            .and_then(|href| final_url.join(href.trim()).ok())
            .filter(|u| !matches!(u.scheme(), "data" | "javascript"))
            .unwrap_or_else(|| final_url.clone());

        let body_selector = scraper::Selector::parse("body").unwrap();
        let content = if let Some(body) = document.select(&body_selector).next() {
            extract_text(body, include_links, &base_url, false)
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

fn too_large(max: usize) -> Error {
    Error::Validation(format!("Content exceeds maximum length of {max} bytes"))
}

/// Resolves `href` against `base`, falling back to the raw value when it cannot be parsed.
fn resolve_href(base: &url::Url, href: &str) -> String {
    base.join(href.trim()).map(String::from).unwrap_or_else(|_| href.to_string())
}

/// Extracts text from an HTML element, optionally preserving links.
///
/// Links are resolved against `base`. `header`/`footer` elements nested inside
/// `main`/`article`/`section` are kept (they hold headings and attribution);
/// page-level ones are dropped. `in_section` tracks that nesting.
fn extract_text(element: scraper::ElementRef, include_links: bool, base: &url::Url, in_section: bool) -> String {
    let mut text = String::new();
    // Remove script and style elements by skipping them
    let skip_tags = ["script", "style", "noscript", "nav"];

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
                if skip_tags.contains(&tag) || (!in_section && matches!(tag, "header" | "footer")) {
                    continue;
                }
                let child_in_section = in_section || matches!(tag, "main" | "article" | "section");

                if let Some(child_ref) = scraper::ElementRef::wrap(node) {
                    let child_text = extract_text(child_ref, include_links, base, child_in_section);

                    if include_links && tag == "a" {
                        if let Some(href) = el.attr("href") {
                            if !child_text.trim().is_empty() {
                                text.push_str(&format!("[{}]({})", child_text.trim(), resolve_href(base, href)));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(html: &str, base: &str) -> String {
        let doc = scraper::Html::parse_document(html);
        let sel = scraper::Selector::parse("body").unwrap();
        let body = doc.select(&sel).next().unwrap();
        extract_text(body, true, &url::Url::parse(base).unwrap(), false)
    }

    #[test]
    fn links_resolve_against_final_url() {
        let out = extract(r#"<body><p><a href="/a/b">x</a> <a href="rel">y</a></p></body>"#, "https://example.com/dir/page");
        assert!(out.contains("[x](https://example.com/a/b)"), "{out}");
        assert!(out.contains("[y](https://example.com/dir/rel)"), "{out}");
    }

    #[test]
    fn page_level_header_footer_dropped_but_article_ones_kept() {
        let html = "<body><header>Site</header><article><header><h1>Title</h1>By Ann</header><p>Body</p><footer>Source X</footer></article><footer>Copyright</footer></body>";
        let out = extract(html, "https://e.com/");
        assert!(!out.contains("Site") && !out.contains("Copyright"), "{out}");
        assert!(out.contains("Title") && out.contains("By Ann") && out.contains("Source X"), "{out}");
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test(flavor = "multi_thread")]
    async fn oversized_body_is_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string(format!("<html><body>{}</body></html>", "a".repeat(500))))
            .mount(&server)
            .await;
        let cfg = WebpageScraperConfig { max_content_length: 100, ..Default::default() };
        let tool = WebpageScraperTool::new(cfg);
        assert!(tool.scrape(&server.uri(), true).await.is_err());

        let tool = WebpageScraperTool::default();
        let r = tool.scrape(&server.uri(), true).await.unwrap();
        assert!(r.content.starts_with('a'));
    }
}
