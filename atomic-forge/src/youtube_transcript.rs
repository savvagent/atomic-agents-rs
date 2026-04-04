use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Fetch the transcript of a YouTube video.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct YouTubeTranscriptInput {
    /// The YouTube video URL or video ID.
    pub video_url: String,
    /// Language code for the transcript (e.g., "en").
    pub language: Option<String>,
}

/// Metadata about a YouTube video.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct VideoMetadata {
    /// The video ID.
    pub id: String,
    /// The video title.
    pub title: String,
    /// The channel name.
    pub channel: String,
}

/// Output of the YouTube transcript tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct YouTubeTranscriptOutput {
    /// The full transcript text.
    pub transcript: String,
    /// Video metadata.
    pub metadata: VideoMetadata,
}

/// Configuration for the YouTube transcript tool.
pub struct YouTubeTranscriptConfig {
    pub component: ComponentConfig,
    pub api_key: String,
}

impl YouTubeTranscriptConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("YouTubeTranscript")
                .with_description("Fetches YouTube video transcripts"),
            api_key: api_key.into(),
        }
    }
}

/// Tool for fetching YouTube video transcripts.
///
/// Uses the YouTube Data API to fetch video metadata and captions.
pub struct YouTubeTranscriptTool {
    config: ComponentConfig,
    api_key: String,
    client: reqwest::Client,
}

impl YouTubeTranscriptTool {
    pub fn new(yt_config: YouTubeTranscriptConfig) -> Self {
        let api_key = if yt_config.api_key.is_empty() {
            std::env::var("YOUTUBE_API_KEY").unwrap_or_default()
        } else {
            yt_config.api_key
        };

        Self {
            config: yt_config.component,
            api_key,
            client: reqwest::Client::new(),
        }
    }

    fn extract_video_id(url: &str) -> Option<String> {
        // Handle various YouTube URL formats
        if url.len() == 11 && !url.contains('/') {
            return Some(url.to_string());
        }

        if let Ok(parsed) = url::Url::parse(url) {
            // youtube.com/watch?v=ID
            if let Some(v) = parsed.query_pairs().find(|(k, _)| k == "v") {
                return Some(v.1.to_string());
            }
            // youtu.be/ID
            let path = parsed.path().trim_start_matches('/');
            if !path.is_empty() && !path.contains('/') {
                return Some(path.to_string());
            }
        }

        None
    }

    async fn fetch_metadata(&self, video_id: &str) -> std::result::Result<VideoMetadata, Error> {
        let url = format!(
            "https://www.googleapis.com/youtube/v3/videos?part=snippet&id={}&key={}",
            video_id, self.api_key
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("YouTube API request failed: {e}")))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse YouTube response: {e}")))?;

        let item = body
            .get("items")
            .and_then(|i| i.get(0))
            .ok_or_else(|| Error::Validation("Video not found".to_string()))?;

        let snippet = item.get("snippet").unwrap_or(&serde_json::Value::Null);

        Ok(VideoMetadata {
            id: video_id.to_string(),
            title: snippet
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            channel: snippet
                .get("channelTitle")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string(),
        })
    }

    async fn fetch_transcript(
        &self,
        video_id: &str,
        _language: Option<&str>,
    ) -> std::result::Result<String, Error> {
        // Fetch captions list
        let url = format!(
            "https://www.googleapis.com/youtube/v3/captions?part=snippet&videoId={}&key={}",
            video_id, self.api_key
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("YouTube captions API failed: {e}")))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::Validation(format!("Failed to parse captions response: {e}")))?;

        let items = body
            .get("items")
            .and_then(|i| i.as_array())
            .ok_or_else(|| Error::Validation("No captions found for this video".to_string()))?;

        if items.is_empty() {
            return Err(Error::Validation("No captions available".to_string()));
        }

        // Get the first caption track ID
        let caption_id = items[0]
            .get("id")
            .and_then(|i| i.as_str())
            .ok_or_else(|| Error::Validation("No caption ID found".to_string()))?;

        // Download the caption track
        let download_url = format!(
            "https://www.googleapis.com/youtube/v3/captions/{}?key={}",
            caption_id, self.api_key
        );

        let transcript = self
            .client
            .get(&download_url)
            .send()
            .await
            .map_err(|e| Error::Validation(format!("Failed to download captions: {e}")))?
            .text()
            .await
            .map_err(|e| Error::Validation(format!("Failed to read captions: {e}")))?;

        Ok(transcript)
    }
}

impl Tool for YouTubeTranscriptTool {
    type Input = YouTubeTranscriptInput;
    type Output = YouTubeTranscriptOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let video_id = Self::extract_video_id(&input.video_url)
            .ok_or_else(|| Error::Validation("Could not extract video ID from URL".to_string()))?;

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let metadata = self.fetch_metadata(&video_id).await?;
                let transcript = self
                    .fetch_transcript(&video_id, input.language.as_deref())
                    .await?;

                Ok(YouTubeTranscriptOutput {
                    transcript,
                    metadata,
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_video_id_full_url() {
        let id = YouTubeTranscriptTool::extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        );
        assert_eq!(id, Some("dQw4w9WgXcQ".to_string()));
    }

    #[test]
    fn test_extract_video_id_short_url() {
        let id =
            YouTubeTranscriptTool::extract_video_id("https://youtu.be/dQw4w9WgXcQ");
        assert_eq!(id, Some("dQw4w9WgXcQ".to_string()));
    }

    #[test]
    fn test_extract_video_id_raw() {
        let id = YouTubeTranscriptTool::extract_video_id("dQw4w9WgXcQ");
        assert_eq!(id, Some("dQw4w9WgXcQ".to_string()));
    }
}
