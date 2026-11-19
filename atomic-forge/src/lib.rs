//! # Atomic Forge
//!
//! Collection of reusable tools for the atomic-agents framework.
//!
//! Each tool implements the `Tool` trait with typed input/output schemas.
//!
//! ## Available Tools
//!
//! - [`calculator`] - Mathematical expression evaluation
//! - [`tavily_search`] - Web search via Tavily API
//! - [`webpage_scraper`] - HTML to text webpage scraping
//! - [`youtube_transcript`] - YouTube video transcript extraction
//! - [`searxng_search`] - Meta-search via SearXNG
//! - [`bocha_search`] - Web search via BoCha API
//! - [`arxiv_search`] - Academic paper search via the arXiv API
//! - [`hackernews_search`] - Hacker News search via Algolia
//! - [`wikipedia_search`] - Wikipedia article search
//! - [`weather`] - Current weather and forecast via Open-Meteo
//! - [`firecrawl_search`] - Web and news search via Firecrawl
//! - `datetime_tool` - Date/time operations (feature `datetime`)
//! - `pdf_reader` - PDF text extraction (feature `pdf-reader`)
//!
//! ## Features
//!
//! - `datetime` - enables `datetime_tool` (adds `chrono` and `chrono-tz`)
//! - `pdf-reader` - enables `pdf_reader` (adds `lopdf`)

pub mod arxiv_search;
pub mod bocha_search;
pub mod calculator;
#[cfg(feature = "datetime")]
pub mod datetime_tool;
pub mod firecrawl_search;
pub mod hackernews_search;
pub mod net_guard;
#[cfg(feature = "pdf-reader")]
pub mod pdf_reader;
pub mod searxng_search;
pub mod tavily_search;
pub mod weather;
pub mod webpage_scraper;
pub mod wikipedia_search;
pub mod youtube_transcript;
