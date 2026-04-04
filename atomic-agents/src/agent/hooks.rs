use std::collections::HashMap;
use std::sync::Arc;

/// Hook events that can be registered with an agent.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum HookEvent {
    /// Fired when structured output parsing fails.
    ParseError,
    /// Fired before sending the completion request (with kwargs).
    CompletionKwargs,
    /// Fired after receiving a completion response.
    CompletionResponse,
    /// Fired on completion errors.
    CompletionError,
    /// Fired on the last retry attempt.
    CompletionLastAttempt,
    /// Fired after token counting completes.
    TokenCounted,
}

impl HookEvent {
    pub fn as_str(&self) -> &str {
        match self {
            HookEvent::ParseError => "parse:error",
            HookEvent::CompletionKwargs => "completion:kwargs",
            HookEvent::CompletionResponse => "completion:response",
            HookEvent::CompletionError => "completion:error",
            HookEvent::CompletionLastAttempt => "completion:last_attempt",
            HookEvent::TokenCounted => "token:counted",
        }
    }
}

/// A hook handler function.
pub type HookHandler = Arc<dyn Fn(&serde_json::Value) + Send + Sync>;

/// Registry for hook handlers.
pub struct HookRegistry {
    handlers: HashMap<HookEvent, Vec<HookHandler>>,
    enabled: bool,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            enabled: true,
        }
    }

    /// Registers a handler for a hook event.
    pub fn register(&mut self, event: HookEvent, handler: HookHandler) {
        self.handlers.entry(event).or_default().push(handler);
    }

    /// Removes a specific handler. Since we use Arc, we compare by pointer equality.
    pub fn unregister(&mut self, event: &HookEvent, handler: &HookHandler) {
        if let Some(handlers) = self.handlers.get_mut(event) {
            handlers.retain(|h| !Arc::ptr_eq(h, handler));
        }
    }

    /// Clears all handlers for a specific event, or all events if None.
    pub fn clear(&mut self, event: Option<&HookEvent>) {
        match event {
            Some(e) => {
                self.handlers.remove(e);
            }
            None => {
                self.handlers.clear();
            }
        }
    }

    /// Dispatches a hook event with the given data.
    ///
    /// Handler exceptions are caught and logged, never propagated.
    pub fn dispatch(&self, event: &HookEvent, data: &serde_json::Value) {
        if !self.enabled {
            return;
        }
        if let Some(handlers) = self.handlers.get(event) {
            for handler in handlers {
                // Catch panics from handlers to avoid disrupting the agent
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    handler(data);
                }));
                if let Err(e) = result {
                    tracing::warn!(
                        event = event.as_str(),
                        "Hook handler panicked: {:?}",
                        e
                    );
                }
            }
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistry")
            .field("enabled", &self.enabled)
            .field(
                "handlers",
                &self
                    .handlers
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.len()))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_register_and_dispatch() {
        let mut registry = HookRegistry::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let handler: HookHandler = Arc::new(move |_data| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        registry.register(HookEvent::ParseError, handler);
        registry.dispatch(&HookEvent::ParseError, &serde_json::json!({}));

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_disable_hooks() {
        let mut registry = HookRegistry::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let handler: HookHandler = Arc::new(move |_data| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        registry.register(HookEvent::ParseError, handler);
        registry.disable();
        registry.dispatch(&HookEvent::ParseError, &serde_json::json!({}));

        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_clear_specific_event() {
        let mut registry = HookRegistry::new();
        let handler: HookHandler = Arc::new(|_| {});

        registry.register(HookEvent::ParseError, handler.clone());
        registry.register(HookEvent::CompletionResponse, handler);

        registry.clear(Some(&HookEvent::ParseError));
        assert!(registry.handlers.get(&HookEvent::ParseError).is_none());
        assert!(registry.handlers.get(&HookEvent::CompletionResponse).is_some());
    }

    #[test]
    fn test_handler_panic_does_not_propagate() {
        let mut registry = HookRegistry::new();
        let handler: HookHandler = Arc::new(|_| {
            panic!("intentional test panic");
        });

        registry.register(HookEvent::ParseError, handler);
        // Should not panic
        registry.dispatch(&HookEvent::ParseError, &serde_json::json!({}));
    }
}
