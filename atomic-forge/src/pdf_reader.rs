use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

use atomic_agents::prelude::*;
use lopdf::{Dictionary, Document, Object};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::net_guard::NetGuard;

fn default_true() -> bool {
    true
}

/// Read a PDF file and extract its text. The source can be a local file path or
/// an HTTP(S) URL. An optional page range limits extraction to specific pages
/// (1-based, e.g. '1', '3-5', '1,3,5-7'). Use this whenever the agent needs the
/// contents of a PDF - papers, reports, datasheets, etc.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct PdfReaderInput {
    /// Local file path (absolute or relative) or HTTP(S) URL of the PDF to read.
    pub source: String,
    /// Optional 1-based page-range expression to limit extraction. Single pages
    /// ('3'), ranges ('1-5'), and comma-separated combinations ('1,3,5-7') are
    /// supported. When omitted, all pages are read.
    #[serde(default)]
    pub page_range: Option<String>,
    /// Whether to include document metadata in the output.
    #[serde(default = "default_true")]
    pub include_metadata: bool,
    /// Whether to extract each page's embedded URI link targets.
    #[serde(default)]
    pub include_links: bool,
}

/// A single extracted PDF page.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct PdfPage {
    /// 1-based page number in the source document.
    pub page_number: u32,
    /// Extracted text for this page.
    pub text: String,
    /// URI annotation targets in page order, when include_links is true.
    #[serde(default)]
    pub links: Vec<String>,
}

/// Metadata fields read from the PDF document info dictionary.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, Default, PartialEq)]
pub struct PdfMetadata {
    /// Document title.
    pub title: Option<String>,
    /// Document author.
    pub author: Option<String>,
    /// Document subject.
    pub subject: Option<String>,
    /// Application that created the original document.
    pub creator: Option<String>,
    /// Application that produced the PDF.
    pub producer: Option<String>,
    /// Raw creation date string from the PDF.
    pub creation_date: Option<String>,
    /// Raw modification date string from the PDF.
    pub modification_date: Option<String>,
}

/// Output of the PdfReaderTool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct PdfReaderOutput {
    /// The source path or URL that was read (echoed back).
    pub source: String,
    /// Concatenated text across all extracted pages, separated by form feeds.
    pub text: String,
    /// Per-page extracted text.
    pub pages: Vec<PdfPage>,
    /// Number of pages extracted (after page_range filtering).
    pub page_count: usize,
    /// Total number of pages in the source document.
    pub total_page_count: usize,
    /// Document metadata, when requested.
    pub metadata: Option<PdfMetadata>,
    /// Error message when the operation failed.
    #[serde(default)]
    pub error: Option<String>,
}

/// Configuration for the PdfReaderTool.
pub struct PdfReaderConfig {
    pub component: ComponentConfig,
    /// User agent for HTTP requests when the source is a URL.
    pub user_agent: String,
    /// HTTP timeout for URL sources.
    pub timeout: Duration,
    /// Maximum allowed PDF size in bytes when downloading from a URL (default 50 MB).
    pub max_size_bytes: u64,
    /// Allow URL sources that resolve to loopback, private, link-local (including
    /// cloud metadata endpoints), CGNAT, multicast or unspecified addresses.
    /// Default `false`, which blocks SSRF through LLM-supplied URLs.
    pub allow_private_hosts: bool,
    /// Restrict local-path sources to files inside this directory. The path is
    /// canonicalized (resolving `..` and symlinks) and must lie within the
    /// canonicalized directory. Default `None`: any readable local file may be read.
    pub allowed_dir: Option<PathBuf>,
}

impl Default for PdfReaderConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("PdfReader")
                .with_description("Extract text and metadata from local or remote PDF files"),
            user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                         (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                .to_string(),
            timeout: Duration::from_secs(30),
            max_size_bytes: 50 * 1024 * 1024,
            allow_private_hosts: false,
            allowed_dir: None,
        }
    }
}

/// Tool for extracting text and metadata from local or remote PDF files.
///
/// Requires the `pdf-reader` feature.
///
/// # Security
///
/// The `source` is supplied by the caller (typically an LLM). URL sources are
/// checked against private/loopback/link-local addresses (see
/// [`PdfReaderConfig::allow_private_hosts`]) and redirects are re-validated.
/// Local paths are **unrestricted by default**, so an agent can read any file the
/// process can read; set [`PdfReaderConfig::allowed_dir`] when the input is untrusted.
/// Only PDF content is parsed, but error messages may reveal whether a path exists.
///
/// # Example
///
/// ```no_run
/// use atomic_agents::prelude::*;
/// use atomic_forge::pdf_reader::*;
///
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let tool = PdfReaderTool::new(PdfReaderConfig::default());
/// let out = tool.run(serde_json::from_str(
///     r#"{"source": "https://arxiv.org/pdf/1706.03762", "page_range": "1-2"}"#,
/// ).unwrap()).unwrap();
/// println!("{}", out.text);
/// # }
/// ```
pub struct PdfReaderTool {
    config: ComponentConfig,
    max_size_bytes: u64,
    user_agent: String,
    timeout: Duration,
    guard: NetGuard,
    allowed_dir: Option<PathBuf>,
}

fn is_url(source: &str) -> bool {
    let lower = source.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

impl PdfReaderTool {
    pub fn new(config: PdfReaderConfig) -> Self {
        Self {
            config: config.component,
            max_size_bytes: config.max_size_bytes,
            user_agent: config.user_agent,
            timeout: config.timeout,
            guard: NetGuard::new(config.allow_private_hosts),
            allowed_dir: config.allowed_dir,
        }
    }

    /// Parse a 1-based page-range expression into a sorted list of page numbers,
    /// clipped to `total_pages`.
    ///
    /// ```
    /// use atomic_forge::pdf_reader::PdfReaderTool;
    /// assert_eq!(PdfReaderTool::parse_page_range("1,3,5-7", 10).unwrap(), vec![1, 3, 5, 6, 7]);
    /// assert_eq!(PdfReaderTool::parse_page_range("8-20", 10).unwrap(), vec![8, 9, 10]);
    /// assert!(PdfReaderTool::parse_page_range("0", 10).is_err());
    /// ```
    pub fn parse_page_range(
        expression: &str,
        total_pages: usize,
    ) -> std::result::Result<Vec<u32>, String> {
        let text = expression.trim();
        if text.is_empty() {
            return Err("page_range cannot be empty when provided.".to_string());
        }
        let total = u32::try_from(total_pages).unwrap_or(u32::MAX);
        let mut result = BTreeSet::new();
        for part in text.split(',') {
            let chunk = part.trim();
            if chunk.is_empty() {
                continue;
            }
            if let Some((a, b)) = chunk.split_once('-') {
                let invalid = || format!("Invalid page range segment '{chunk}'.");
                let start: u32 = a.trim().parse().map_err(|_| invalid())?;
                let end: u32 = b.trim().parse().map_err(|_| invalid())?;
                if start < 1 || end < 1 || end < start {
                    return Err(invalid());
                }
                result.extend(start..=end.min(total));
            } else {
                let page: i64 = chunk
                    .parse()
                    .map_err(|_| format!("Invalid page number '{chunk}'."))?;
                if page < 1 {
                    return Err(format!("Page numbers must be >= 1 (got '{chunk}')."));
                }
                if page <= i64::from(total) {
                    result.insert(page as u32);
                }
            }
        }
        if result.is_empty() {
            return Err(format!(
                "No pages selected by '{expression}' (document has {total_pages} pages)."
            ));
        }
        Ok(result.into_iter().collect())
    }

    async fn download(&self, url: &str) -> std::result::Result<Vec<u8>, String> {
        let mut response = self
            .guard
            .get(url, &self.user_agent, "application/pdf", self.timeout)
            .await
            .map_err(|e| format!("Failed to download PDF: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("Failed to download PDF: HTTP {status}"));
        }
        if let Some(len) = response.content_length()
            && len > self.max_size_bytes
        {
            return Err(format!(
                "PDF too large: declared {len} bytes (limit {}).",
                self.max_size_bytes
            ));
        }
        let mut data = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| format!("Failed to download PDF: {e}"))?
        {
            data.extend_from_slice(&chunk);
            if data.len() as u64 > self.max_size_bytes {
                return Err(format!(
                    "PDF too large: exceeded {} bytes during download.",
                    self.max_size_bytes
                ));
            }
        }
        Ok(data)
    }

    async fn load_bytes(&self, source: &str) -> std::result::Result<Vec<u8>, String> {
        if is_url(source) {
            return self.download(source).await;
        }
        let path = match source.strip_prefix("~/") {
            Some(rest) => std::env::var_os("HOME")
                .map(|home| std::path::PathBuf::from(home).join(rest))
                .unwrap_or_else(|| source.into()),
            None => source.into(),
        };
        if !path.exists() {
            return Err(format!("PDF not found at '{source}'."));
        }
        // With a sandbox, only ever open the canonical path that was validated, so a
        // symlink swapped in after the check cannot redirect the read.
        let target = if let Some(dir) = &self.allowed_dir {
            let dir = dir
                .canonicalize()
                .map_err(|e| format!("Allowed directory is not accessible: {e}"))?;
            let real = path
                .canonicalize()
                .map_err(|e| format!("Failed to resolve '{source}': {e}"))?;
            if !real.starts_with(&dir) {
                return Err(format!(
                    "Access denied: '{source}' is outside the allowed directory."
                ));
            }
            Some((dir, real))
        } else {
            None
        };
        let open_path = target.as_ref().map_or(&path, |(_, real)| real);
        let mut file = tokio::fs::File::open(open_path)
            .await
            .map_err(|e| format!("Failed to read '{source}': {e}"))?;
        let meta = file
            .metadata()
            .await
            .map_err(|e| format!("Failed to read '{source}': {e}"))?;
        if !meta.is_file() {
            return Err(format!("'{source}' is not a file."));
        }
        // Re-validate through the open handle: the file we actually hold must still
        // live inside the allowed directory.
        #[cfg(target_os = "linux")]
        if let Some((dir, _)) = &target {
            use std::os::fd::AsRawFd;
            let opened = std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))
                .map_err(|e| format!("Failed to verify '{source}': {e}"))?;
            if !opened.starts_with(dir) {
                return Err(format!(
                    "Access denied: '{source}' is outside the allowed directory."
                ));
            }
        }
        let mut data = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut file, &mut data)
            .await
            .map_err(|e| format!("Failed to read '{source}': {e}"))?;
        Ok(data)
    }

    fn info_dictionary(doc: &Document) -> Option<&Dictionary> {
        let info = doc.trailer.get(b"Info").ok()?;
        let (_, obj) = doc.dereference(info).ok()?;
        obj.as_dict().ok()
    }

    /// Read the document info dictionary.
    pub fn read_metadata(doc: &Document) -> PdfMetadata {
        let info = Self::info_dictionary(doc);
        let get = |key: &[u8]| -> Option<String> {
            let info = info?;
            let (_, obj) = doc.dereference(info.get(key).ok()?).ok()?;
            let text = match obj {
                Object::String(..) => lopdf::decode_text_string(obj).ok()?,
                Object::Name(n) => String::from_utf8_lossy(n).into_owned(),
                _ => return None,
            };
            (!text.is_empty()).then_some(text)
        };
        PdfMetadata {
            title: get(b"Title"),
            author: get(b"Author"),
            subject: get(b"Subject"),
            creator: get(b"Creator"),
            producer: get(b"Producer"),
            creation_date: get(b"CreationDate"),
            modification_date: get(b"ModDate"),
        }
    }

    /// Collect the URI targets of a page's link annotations, in page order.
    pub fn page_links(doc: &Document, page_id: lopdf::ObjectId) -> Vec<String> {
        let Ok(annotations) = doc.get_page_annotations(page_id) else {
            return Vec::new();
        };
        annotations
            .into_iter()
            .filter(|a| {
                a.get(b"Subtype")
                    .ok()
                    .and_then(|o| o.as_name().ok())
                    .is_some_and(|n| n == b"Link")
            })
            .filter_map(|a| {
                let (_, action) = doc.dereference(a.get(b"A").ok()?).ok()?;
                let action = action.as_dict().ok()?;
                if action.get(b"S").ok()?.as_name().ok()? != b"URI" {
                    return None;
                }
                let (_, uri) = doc.dereference(action.get(b"URI").ok()?).ok()?;
                let text = lopdf::decode_text_string(uri).ok()?;
                (!text.is_empty()).then_some(text)
            })
            .collect()
    }

    /// Extract text from PDF bytes according to the input options.
    pub fn read_bytes(
        data: &[u8],
        input: &PdfReaderInput,
    ) -> std::result::Result<PdfReaderOutput, String> {
        let doc = Document::load_mem(data).map_err(|e| format!("Failed to parse PDF: {e}"))?;
        if doc.is_encrypted() {
            return Err(
                "PDF is encrypted and could not be decrypted without a password.".to_string(),
            );
        }
        let page_ids = doc.get_pages();
        let total = page_ids.len();
        let numbers = match input.page_range.as_deref() {
            Some(expr) => Self::parse_page_range(expr, total)?,
            None => page_ids.keys().copied().collect(),
        };

        let mut pages = Vec::with_capacity(numbers.len());
        for number in numbers {
            let text = doc
                .extract_text(&[number])
                .map_err(|e| format!("Failed to extract text from page {number}: {e}"))?;
            let links = if input.include_links {
                page_ids
                    .get(&number)
                    .map(|id| Self::page_links(&doc, *id))
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            pages.push(PdfPage {
                page_number: number,
                text,
                links,
            });
        }

        let text = pages
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\u{c}\n");
        Ok(PdfReaderOutput {
            source: input.source.clone(),
            text,
            page_count: pages.len(),
            pages,
            total_page_count: total,
            metadata: input.include_metadata.then(|| Self::read_metadata(&doc)),
            error: None,
        })
    }

    async fn read(&self, input: &PdfReaderInput) -> std::result::Result<PdfReaderOutput, String> {
        let data = self.load_bytes(&input.source).await?;
        Self::read_bytes(&data, input)
    }
}

impl Tool for PdfReaderTool {
    type Input = PdfReaderInput;
    type Output = PdfReaderOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let outcome = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.read(&input))
        });
        Ok(outcome.unwrap_or_else(|error| PdfReaderOutput {
            source: input.source.clone(),
            text: String::new(),
            pages: Vec::new(),
            page_count: 0,
            total_page_count: 0,
            metadata: None,
            error: Some(error),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::{Content, Operation};
    use lopdf::{Stream, dictionary};
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Build a PDF with one page per entry of `texts`. The first page carries a link annotation.
    fn sample_pdf(texts: &[&str]) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Courier",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });
        let mut kids = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 24.into()]),
                    Operation::new("Td", vec![100.into(), 600.into()]),
                    Operation::new("Tj", vec![Object::string_literal(*text)]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            let mut page = dictionary! {
                "Type" => "Page", "Parent" => pages_id, "Contents" => content_id,
            };
            if i == 0 {
                let link = doc.add_object(dictionary! {
                    "Type" => "Annot", "Subtype" => "Link",
                    "Rect" => vec![0.into(), 0.into(), 100.into(), 20.into()],
                    "A" => dictionary! {
                        "S" => "URI",
                        "URI" => Object::string_literal("https://example.com/doc"),
                    },
                });
                page.set("Annots", vec![Object::Reference(link)]);
            }
            kids.push(Object::Reference(doc.add_object(page)));
        }
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages", "Kids" => kids, "Count" => texts.len() as i64,
                "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let info_id = doc.add_object(dictionary! {
            "Title" => Object::string_literal("Test Title"),
            "Author" => Object::string_literal("Ada"),
            "Producer" => Object::string_literal("lopdf"),
        });
        doc.trailer.set("Info", info_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    fn input(source: &str) -> PdfReaderInput {
        serde_json::from_value(serde_json::json!({ "source": source })).unwrap()
    }

    #[test]
    fn schema_metadata() {
        assert!(PdfReaderInput::schema_description().contains("PDF"));
        let schema = serde_json::to_value(PdfReaderInput::generate_schema()).unwrap();
        assert!(schema["properties"].get("page_range").is_some());
        let out = serde_json::to_value(PdfReaderOutput::generate_schema()).unwrap();
        assert!(out["properties"].get("total_page_count").is_some());
    }

    #[test]
    fn defaults_apply() {
        let i = input("x.pdf");
        assert!(i.include_metadata);
        assert!(!i.include_links);
        assert!(i.page_range.is_none());
    }

    #[test]
    fn page_range_parsing() {
        let p = PdfReaderTool::parse_page_range;
        assert_eq!(p("3", 5).unwrap(), vec![3]);
        assert_eq!(p("1-3, 5", 5).unwrap(), vec![1, 2, 3, 5]);
        assert_eq!(p("2-99", 4).unwrap(), vec![2, 3, 4]);
        assert_eq!(p("3,3,1", 5).unwrap(), vec![1, 3]);
        assert!(p("", 5).is_err());
        assert!(p("abc", 5).is_err());
        assert!(p("5-3", 5).is_err());
        assert!(p("0-2", 5).is_err());
        assert!(p("-1", 5).is_err());
        assert!(p("9", 5).unwrap_err().contains("No pages selected"));
    }

    #[test]
    fn extracts_text_pages_links_and_metadata() {
        let pdf = sample_pdf(&["Hello PDF", "Second page"]);
        let mut i = input("mem");
        i.include_links = true;
        let out = PdfReaderTool::read_bytes(&pdf, &i).unwrap();
        assert_eq!(out.total_page_count, 2);
        assert_eq!(out.page_count, 2);
        assert!(out.pages[0].text.contains("Hello PDF"));
        assert!(out.pages[1].text.contains("Second page"));
        assert!(out.text.contains('\u{c}'));
        assert_eq!(out.pages[0].links, vec!["https://example.com/doc"]);
        assert!(out.pages[1].links.is_empty());
        let meta = out.metadata.unwrap();
        assert_eq!(meta.title.as_deref(), Some("Test Title"));
        assert_eq!(meta.author.as_deref(), Some("Ada"));
        assert_eq!(meta.subject, None);
    }

    #[test]
    fn page_range_limits_extraction() {
        let pdf = sample_pdf(&["One", "Two", "Three"]);
        let mut i = input("mem");
        i.page_range = Some("2-3".into());
        i.include_metadata = false;
        let out = PdfReaderTool::read_bytes(&pdf, &i).unwrap();
        assert_eq!(out.page_count, 2);
        assert_eq!(out.total_page_count, 3);
        assert_eq!(out.pages[0].page_number, 2);
        assert!(out.metadata.is_none());
        assert!(out.pages[0].links.is_empty());
    }

    #[test]
    fn garbage_bytes_are_an_error() {
        let err = PdfReaderTool::read_bytes(b"not a pdf at all", &input("mem")).unwrap_err();
        assert!(err.contains("Failed to parse PDF"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn missing_file_sets_error() {
        let tool = PdfReaderTool::new(PdfReaderConfig::default());
        let out = tool.run(input("/definitely/not/here.pdf")).unwrap();
        assert!(out.error.unwrap().contains("PDF not found"));
        assert_eq!(out.page_count, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn directory_sets_error() {
        let tool = PdfReaderTool::new(PdfReaderConfig::default());
        let out = tool.run(input(env!("CARGO_MANIFEST_DIR"))).unwrap();
        assert!(out.error.unwrap().contains("is not a file"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reads_local_file() {
        let path =
            std::env::temp_dir().join(format!("atomic_forge_pdf_{}.pdf", std::process::id()));
        std::fs::write(&path, sample_pdf(&["From disk"])).unwrap();
        let tool = PdfReaderTool::new(PdfReaderConfig::default());
        let out = tool.run(input(path.to_str().unwrap())).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(out.error.is_none(), "{:?}", out.error);
        assert!(out.text.contains("From disk"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn downloads_from_url_and_enforces_limit() {
        let server = MockServer::start().await;
        let pdf = sample_pdf(&["Remote"]);
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(pdf.clone()))
            .mount(&server)
            .await;
        let tool = PdfReaderTool::new(PdfReaderConfig {
            allow_private_hosts: true,
            ..Default::default()
        });
        let out = tool.run(input(&format!("{}/a.pdf", server.uri()))).unwrap();
        assert!(out.error.is_none(), "{:?}", out.error);
        assert!(out.text.contains("Remote"));

        let small = PdfReaderTool::new(PdfReaderConfig {
            allow_private_hosts: true,
            max_size_bytes: 10,
            ..Default::default()
        });
        let out = small
            .run(input(&format!("{}/a.pdf", server.uri())))
            .unwrap();
        assert!(out.error.unwrap().contains("too large"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn http_error_sets_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        let tool = PdfReaderTool::new(PdfReaderConfig {
            allow_private_hosts: true,
            ..Default::default()
        });
        let out = tool.run(input(&format!("{}/a.pdf", server.uri()))).unwrap();
        assert!(out.error.unwrap().contains("404"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn private_hosts_blocked_by_default() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(sample_pdf(&["x"])))
            .expect(0)
            .mount(&server)
            .await;
        let tool = PdfReaderTool::new(PdfReaderConfig::default());
        let out = tool.run(input(&format!("{}/a.pdf", server.uri()))).unwrap();
        assert!(out.error.unwrap().contains("non-public"));
        let meta = tool
            .run(input("http://169.254.169.254/latest/meta-data"))
            .unwrap();
        assert!(meta.error.unwrap().contains("non-public"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn allowed_dir_blocks_escape() {
        let base =
            std::env::temp_dir().join(format!("atomic_forge_pdf_dir_{}", std::process::id()));
        let allowed = base.join("allowed");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::write(allowed.join("ok.pdf"), sample_pdf(&["Inside"])).unwrap();
        std::fs::write(base.join("secret.pdf"), sample_pdf(&["Outside"])).unwrap();
        let tool = PdfReaderTool::new(PdfReaderConfig {
            allowed_dir: Some(allowed.clone()),
            ..Default::default()
        });
        let ok = tool
            .run(input(allowed.join("ok.pdf").to_str().unwrap()))
            .unwrap();
        assert!(ok.error.is_none(), "{:?}", ok.error);
        let escape = format!("{}/../secret.pdf", allowed.display());
        let out = tool.run(input(&escape)).unwrap();
        assert!(out.error.unwrap().contains("outside the allowed directory"));
        std::fs::remove_dir_all(&base).ok();
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn allowed_dir_blocks_symlink_escape() {
        let base =
            std::env::temp_dir().join(format!("atomic_forge_pdf_link_{}", std::process::id()));
        let allowed = base.join("allowed");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::write(base.join("secret.pdf"), sample_pdf(&["Outside"])).unwrap();
        std::os::unix::fs::symlink(base.join("secret.pdf"), allowed.join("link.pdf")).unwrap();
        std::os::unix::fs::symlink(&base, allowed.join("linkdir")).unwrap();
        let tool = PdfReaderTool::new(PdfReaderConfig {
            allowed_dir: Some(allowed.clone()),
            ..Default::default()
        });
        for name in ["link.pdf", "linkdir/secret.pdf"] {
            let out = tool
                .run(input(allowed.join(name).to_str().unwrap()))
                .unwrap();
            assert!(
                out.error.unwrap().contains("outside the allowed directory"),
                "{name}"
            );
        }
        // A directory inside the sandbox is not a regular file.
        std::fs::create_dir(allowed.join("sub.pdf")).unwrap();
        let out = tool
            .run(input(allowed.join("sub.pdf").to_str().unwrap()))
            .unwrap();
        assert!(out.error.unwrap().contains("not a file"));
        std::fs::remove_dir_all(&base).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires network access"]
    async fn live_download() {
        let tool = PdfReaderTool::new(PdfReaderConfig::default());
        let mut i = input("https://arxiv.org/pdf/1706.03762");
        i.page_range = Some("1".into());
        let out = tool.run(i).unwrap();
        assert!(out.error.is_none(), "{:?}", out.error);
        assert!(!out.text.is_empty());
    }
}
