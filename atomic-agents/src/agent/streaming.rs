use std::marker::PhantomData;

use crate::schema::IOSchema;

/// A partial output during streaming, containing accumulated raw JSON
/// and a best-effort parse attempt.
#[derive(Clone, Debug)]
pub struct PartialOutput<O: IOSchema> {
    /// The raw JSON string accumulated so far.
    pub raw: String,
    /// Best-effort partial parse as a JSON value.
    pub partial_value: Option<serde_json::Value>,
    /// Successfully parsed output (only set when the JSON is complete and valid).
    pub parsed: Option<O>,
    _phantom: PhantomData<O>,
}

impl<O: IOSchema> PartialOutput<O> {
    pub(crate) fn new(raw: String) -> Self {
        // Try to parse as the full output type first
        let parsed = serde_json::from_str::<O>(&raw).ok();
        // If full parse fails, try as generic JSON value
        let partial_value = if parsed.is_none() {
            serde_json::from_str::<serde_json::Value>(&raw).ok()
        } else {
            None
        };

        Self {
            raw,
            partial_value,
            parsed,
            _phantom: PhantomData,
        }
    }

    /// Returns true if the output has been fully parsed into the target type.
    pub fn is_complete(&self) -> bool {
        self.parsed.is_some()
    }
}
