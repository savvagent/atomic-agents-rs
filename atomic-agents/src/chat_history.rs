use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::schema::IOSchema;

/// Message role in a conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    /// Provider-specific role (e.g., "model" for Gemini, "developer" for OpenAI).
    #[serde(untagged)]
    Custom(String),
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A content part within a message, supporting multimodal content.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
    #[serde(rename = "image_base64")]
    ImageBase64 { data: String, media_type: String },
    #[serde(rename = "audio_base64")]
    AudioBase64 { data: String, media_type: String },
    /// A video reference sent as an OpenAI-compatible `video_url` content part.
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrlContent },
}

/// Video reference for providers that accept `video_url` content parts
/// (mirrors upstream's `VideoURL`).
///
/// Optional parameters that are not set are omitted when serialized.
///
/// ```
/// use atomic_agents::chat_history::VideoUrlContent;
///
/// let video = VideoUrlContent::new("https://example.com/clip.mp4").with_fps(2.0);
/// let json = serde_json::to_value(&video).unwrap();
/// assert_eq!(json["fps"], 2.0);
/// assert!(json.get("detail").is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoUrlContent {
    /// HTTP(S) or `data:` URL of the video.
    pub url: String,
    /// Frame sampling rate, for providers that accept it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    /// Detail level, for providers that accept it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl VideoUrlContent {
    /// Creates a video reference with only a URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            fps: None,
            detail: None,
        }
    }

    /// Sets the frame sampling rate.
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Sets the detail level.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageUrlContent {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Message content, either plain text or multimodal parts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn text(s: impl Into<String>) -> Self {
        MessageContent::Text(s.into())
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
            _ => None,
        }
    }
}

/// A single message in the conversation history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// Manages conversation history for an agent.
///
/// Supports turn-based tracking, message overflow management, and serialization.
/// Mirrors Python's `ChatHistory` class.
#[derive(Clone, Debug)]
pub struct ChatHistory {
    history: Vec<Message>,
    max_messages: Option<usize>,
    current_turn_id: Option<String>,
}

impl ChatHistory {
    /// Creates a new empty chat history.
    pub fn new(max_messages: Option<usize>) -> Self {
        Self {
            history: Vec::new(),
            max_messages,
            current_turn_id: None,
        }
    }

    /// Generates a new turn ID (UUID v4) for grouping related messages.
    pub fn initialize_turn(&mut self) -> String {
        let turn_id = Uuid::new_v4().to_string();
        self.current_turn_id = Some(turn_id.clone());
        turn_id
    }

    /// Returns the current turn ID, if any.
    pub fn current_turn_id(&self) -> Option<&str> {
        self.current_turn_id.as_deref()
    }

    /// Adds a message from an IOSchema value, serializing it to JSON.
    pub fn add_message<S: IOSchema>(&mut self, role: Role, content: &S) {
        let json = serde_json::to_string(content).expect("IOSchema serialization should not fail");
        self.add_raw_message(role, MessageContent::Text(json));
    }

    /// Adds a message with raw content (already formatted).
    pub fn add_raw_message(&mut self, role: Role, content: MessageContent) {
        let message = Message {
            role,
            content,
            turn_id: self.current_turn_id.clone(),
        };
        self.history.push(message);
        self.manage_overflow();
    }

    /// Returns the history as a list of role/content dicts suitable for LLM API calls.
    pub fn get_history(&self) -> Vec<serde_json::Value> {
        self.history
            .iter()
            .map(|msg| {
                let content = match &msg.content {
                    MessageContent::Text(s) => serde_json::Value::String(s.clone()),
                    MessageContent::Parts(parts) => serde_json::to_value(parts).unwrap(),
                };
                serde_json::json!({
                    "role": msg.role.as_str(),
                    "content": content,
                })
            })
            .collect()
    }

    /// Returns the number of messages.
    pub fn message_count(&self) -> usize {
        self.history.len()
    }

    /// Removes all messages with the given turn ID.
    ///
    /// If the deleted turn was the current turn, the current turn ID moves to
    /// the turn of the last remaining message, or to `None` when the history
    /// is empty (matching upstream `delete_turn_id`). Later messages are then
    /// never filed under the deleted, orphaned ID.
    pub fn delete_turn(&mut self, turn_id: &str) {
        self.history
            .retain(|msg| msg.turn_id.as_deref() != Some(turn_id));
        if self.current_turn_id.as_deref() == Some(turn_id) {
            self.current_turn_id = self.history.last().and_then(|m| m.turn_id.clone());
        }
    }

    /// Deep-copies this chat history.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Serializes the entire history to a JSON string.
    pub fn dump(&self) -> String {
        let data = SerializedHistory {
            messages: self.history.clone(),
            max_messages: self.max_messages,
            current_turn_id: self.current_turn_id.clone(),
        };
        serde_json::to_string(&data).expect("History serialization should not fail")
    }

    /// Restores history from a JSON string produced by `dump()`.
    ///
    /// The snapshot is fully parsed and validated before any state is replaced,
    /// so a failed load leaves the existing history untouched. The restored
    /// `max_messages` capacity is enforced on the restored messages (oldest
    /// messages are dropped if the snapshot holds more than its capacity).
    pub fn load(&mut self, serialized: &str) -> std::result::Result<(), serde_json::Error> {
        let data: SerializedHistory = serde_json::from_str(serialized)?;
        self.history = data.messages;
        self.max_messages = data.max_messages;
        self.current_turn_id = data.current_turn_id;
        self.manage_overflow();
        Ok(())
    }

    /// Returns the distinct turn IDs in the order they first appear (oldest first).
    pub fn turn_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for id in self.history.iter().filter_map(|m| m.turn_id.as_ref()) {
            if !ids.contains(id) {
                ids.push(id.clone());
            }
        }
        ids
    }

    /// Clears all messages and resets the turn ID.
    pub fn clear(&mut self) {
        self.history.clear();
        self.current_turn_id = None;
    }

    /// Returns a reference to the raw messages.
    pub fn messages(&self) -> &[Message] {
        &self.history
    }

    fn manage_overflow(&mut self) {
        if let Some(max) = self.max_messages {
            while self.history.len() > max {
                self.history.remove(0);
            }
        }
    }
}

/// Public contract of a chat history, so custom or persistent memory stores can
/// back an agent (mirrors upstream's `BaseChatHistory`).
///
/// [`ChatHistory`] is the built-in in-memory implementation. An agent snapshots
/// its history with [`BaseChatHistory::boxed_copy`] at construction and restores
/// it in `reset_history()`, so implementations that carry extra state (a
/// database handle, a session id) must return their own type from it.
///
/// ```
/// use atomic_agents::chat_history::{BaseChatHistory, ChatHistory, MessageContent, Role};
///
/// let mut history: Box<dyn BaseChatHistory> = Box::new(ChatHistory::new(Some(2)));
/// history.add_raw_message(Role::User, MessageContent::text("hi"));
/// assert_eq!(history.message_count(), 1);
/// ```
pub trait BaseChatHistory: Send + Sync {
    /// Starts a new turn, returning its freshly generated ID.
    fn initialize_turn(&mut self) -> String;
    /// Returns the current turn ID, if a turn has been started.
    fn current_turn_id(&self) -> Option<&str>;
    /// Adds a message, tagging it with the current turn ID.
    fn add_raw_message(&mut self, role: Role, content: MessageContent);
    /// Returns the history as role/content values suitable for LLM API calls.
    fn get_history(&self) -> Vec<serde_json::Value>;
    /// Returns the number of messages.
    fn message_count(&self) -> usize;
    /// Returns the distinct turn IDs, oldest first.
    fn turn_ids(&self) -> Vec<String>;
    /// Removes all messages with the given turn ID.
    fn delete_turn(&mut self, turn_id: &str);
    /// Serializes the history to a JSON string accepted by `load`.
    fn dump(&self) -> String;
    /// Replaces the history from a string produced by `dump`; must leave the
    /// existing state untouched on failure.
    fn load(&mut self, serialized: &str) -> std::result::Result<(), serde_json::Error>;
    /// Clears all messages and resets the turn ID.
    fn clear(&mut self);
    /// Creates an independent copy of this history.
    fn boxed_copy(&self) -> Box<dyn BaseChatHistory>;
}

impl dyn BaseChatHistory {
    /// Adds a message from an `IOSchema` value, serializing it to JSON.
    pub fn add_message<S: IOSchema>(&mut self, role: Role, content: &S) {
        let json = serde_json::to_string(content).expect("IOSchema serialization should not fail");
        self.add_raw_message(role, MessageContent::Text(json));
    }
}

impl BaseChatHistory for ChatHistory {
    fn initialize_turn(&mut self) -> String {
        ChatHistory::initialize_turn(self)
    }
    fn current_turn_id(&self) -> Option<&str> {
        ChatHistory::current_turn_id(self)
    }
    fn add_raw_message(&mut self, role: Role, content: MessageContent) {
        ChatHistory::add_raw_message(self, role, content)
    }
    fn get_history(&self) -> Vec<serde_json::Value> {
        ChatHistory::get_history(self)
    }
    fn message_count(&self) -> usize {
        ChatHistory::message_count(self)
    }
    fn turn_ids(&self) -> Vec<String> {
        ChatHistory::turn_ids(self)
    }
    fn delete_turn(&mut self, turn_id: &str) {
        ChatHistory::delete_turn(self, turn_id)
    }
    fn dump(&self) -> String {
        ChatHistory::dump(self)
    }
    fn load(&mut self, serialized: &str) -> std::result::Result<(), serde_json::Error> {
        ChatHistory::load(self, serialized)
    }
    fn clear(&mut self) {
        ChatHistory::clear(self)
    }
    fn boxed_copy(&self) -> Box<dyn BaseChatHistory> {
        Box::new(self.clone())
    }
}

impl Default for ChatHistory {
    fn default() -> Self {
        Self::new(None)
    }
}

#[derive(Serialize, Deserialize)]
struct SerializedHistory {
    messages: Vec<Message>,
    max_messages: Option<usize>,
    current_turn_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::BasicChatInputSchema;

    #[test]
    fn test_add_and_get_messages() {
        let mut history = ChatHistory::new(None);
        let input = BasicChatInputSchema {
            chat_message: "Hello".to_string(),
        };
        history.add_message(Role::User, &input);
        assert_eq!(history.message_count(), 1);

        let msgs = history.get_history();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn test_turn_tracking() {
        let mut history = ChatHistory::new(None);
        let turn_id = history.initialize_turn();
        assert!(!turn_id.is_empty());
        assert_eq!(history.current_turn_id(), Some(turn_id.as_str()));

        let input = BasicChatInputSchema {
            chat_message: "test".to_string(),
        };
        history.add_message(Role::User, &input);
        assert_eq!(
            history.messages()[0].turn_id.as_deref(),
            Some(turn_id.as_str())
        );
    }

    #[test]
    fn test_max_messages_overflow() {
        let mut history = ChatHistory::new(Some(2));
        for i in 0..5 {
            history.add_raw_message(Role::User, MessageContent::text(format!("msg {i}")));
        }
        assert_eq!(history.message_count(), 2);
        // Should have the last 2 messages
        let msgs = history.get_history();
        assert_eq!(msgs[0]["content"], "msg 3");
        assert_eq!(msgs[1]["content"], "msg 4");
    }

    #[test]
    fn test_delete_turn() {
        let mut history = ChatHistory::new(None);
        let turn = history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("a"));
        history.add_raw_message(Role::Assistant, MessageContent::text("b"));
        assert_eq!(history.message_count(), 2);

        history.delete_turn(&turn);
        assert_eq!(history.message_count(), 0);
    }

    #[test]
    fn test_dump_and_load() {
        let mut history = ChatHistory::new(Some(10));
        history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("hello"));
        history.add_raw_message(Role::Assistant, MessageContent::text("hi"));

        let serialized = history.dump();
        let mut restored = ChatHistory::new(None);
        restored.load(&serialized).unwrap();

        assert_eq!(restored.message_count(), 2);
        assert_eq!(restored.max_messages, Some(10));
    }

    #[test]
    fn test_multimodal_content() {
        let mut history = ChatHistory::new(None);
        history.add_raw_message(
            Role::User,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What's in this image?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlContent {
                        url: "https://example.com/img.png".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
        );
        assert_eq!(history.message_count(), 1);
        let msgs = history.get_history();
        assert_eq!(msgs[0]["role"], "user");
        assert!(msgs[0]["content"].is_array());
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(Role::Custom("model".to_string()).as_str(), "model");
    }

    #[test]
    fn test_load_enforces_capacity() {
        let mut source = ChatHistory::new(None);
        for i in 0..5 {
            source.add_raw_message(Role::User, MessageContent::text(format!("m{i}")));
        }
        // Hand-edit the snapshot so its capacity is smaller than its message count.
        let mut snapshot: serde_json::Value = serde_json::from_str(&source.dump()).unwrap();
        snapshot["max_messages"] = serde_json::json!(2);

        let mut restored = ChatHistory::new(None);
        restored.load(&snapshot.to_string()).unwrap();
        assert_eq!(restored.message_count(), 2);
        assert_eq!(restored.get_history()[0]["content"], "m3");
    }

    #[test]
    fn test_failed_load_preserves_existing_state() {
        let mut history = ChatHistory::new(Some(5));
        history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("keep me"));

        for bad in [
            "not json",
            r#"{"messages": "nope", "max_messages": null, "current_turn_id": null}"#,
            r#"{"messages": [], "max_messages": -1, "current_turn_id": null}"#,
            r#"{"messages": [], "max_messages": null, "current_turn_id": 7}"#,
        ] {
            assert!(history.load(bad).is_err(), "{bad}");
        }
        assert_eq!(history.message_count(), 1);
        assert_eq!(history.get_history()[0]["content"], "keep me");
        assert_eq!(history.max_messages, Some(5));
        assert!(history.current_turn_id().is_some());
    }

    #[test]
    fn test_delete_current_turn_resets_current_turn_id() {
        let mut history = ChatHistory::new(None);
        let first = history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("a"));
        let second = history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("b"));

        history.delete_turn(&second);
        assert_eq!(history.current_turn_id(), Some(first.as_str()));
        history.add_raw_message(Role::User, MessageContent::text("c"));
        assert_eq!(
            history.messages()[1].turn_id.as_deref(),
            Some(first.as_str())
        );

        // Deleting a non-current turn leaves the current ID alone.
        let third = history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("d"));
        history.delete_turn(&first);
        assert_eq!(history.current_turn_id(), Some(third.as_str()));

        history.delete_turn(&third);
        assert_eq!(history.current_turn_id(), None);
    }

    #[test]
    fn test_turn_ids_ordered_and_distinct() {
        let mut history = ChatHistory::new(None);
        let a = history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("1"));
        history.add_raw_message(Role::Assistant, MessageContent::text("2"));
        let b = history.initialize_turn();
        history.add_raw_message(Role::User, MessageContent::text("3"));
        assert_eq!(history.turn_ids(), vec![a, b]);
    }

    #[test]
    fn test_video_url_part_serialization() {
        let mut history = ChatHistory::new(None);
        history.add_raw_message(
            Role::User,
            MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrlContent::new("https://example.com/v.mp4").with_fps(1.5),
            }]),
        );
        let msgs = history.get_history();
        let part = &msgs[0]["content"][0];
        assert_eq!(part["type"], "video_url");
        assert_eq!(part["video_url"]["url"], "https://example.com/v.mp4");
        assert_eq!(part["video_url"]["fps"], 1.5);
        assert!(part["video_url"].get("detail").is_none());
    }

    #[test]
    fn test_base_chat_history_trait_object() {
        let mut history: Box<dyn BaseChatHistory> = Box::new(ChatHistory::new(None));
        let turn = history.initialize_turn();
        history.add_message(
            Role::User,
            &BasicChatInputSchema {
                chat_message: "x".into(),
            },
        );
        let copy = history.boxed_copy();
        history.delete_turn(&turn);
        assert_eq!(history.message_count(), 0);
        assert_eq!(copy.message_count(), 1);
    }
}
