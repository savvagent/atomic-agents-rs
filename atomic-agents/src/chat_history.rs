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
    ImageUrl {
        image_url: ImageUrlContent,
    },
    #[serde(rename = "image_base64")]
    ImageBase64 {
        data: String,
        media_type: String,
    },
    #[serde(rename = "audio_base64")]
    AudioBase64 {
        data: String,
        media_type: String,
    },
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
    pub fn delete_turn(&mut self, turn_id: &str) {
        self.history.retain(|msg| {
            msg.turn_id.as_deref() != Some(turn_id)
        });
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
    pub fn load(&mut self, serialized: &str) -> std::result::Result<(), serde_json::Error> {
        let data: SerializedHistory = serde_json::from_str(serialized)?;
        self.history = data.messages;
        self.max_messages = data.max_messages;
        self.current_turn_id = data.current_turn_id;
        Ok(())
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
        assert_eq!(history.messages()[0].turn_id.as_deref(), Some(turn_id.as_str()));
    }

    #[test]
    fn test_max_messages_overflow() {
        let mut history = ChatHistory::new(Some(2));
        for i in 0..5 {
            history.add_raw_message(
                Role::User,
                MessageContent::text(format!("msg {i}")),
            );
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
}
