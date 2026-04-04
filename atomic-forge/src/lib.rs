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

pub mod calculator;
pub mod tavily_search;
pub mod webpage_scraper;
pub mod youtube_transcript;
pub mod searxng_search;
pub mod bocha_search;
