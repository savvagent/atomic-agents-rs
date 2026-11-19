use indexmap::IndexMap;

/// Trait for providing dynamic context to the system prompt at runtime.
///
/// Implement this trait to inject context that changes between agent calls,
/// such as search results, database lookups, or user preferences.
///
/// Mirrors Python's `BaseDynamicContextProvider`.
pub trait ContextProvider: Send + Sync {
    /// Display title for this context section in the system prompt.
    fn title(&self) -> &str;

    /// Returns the context information as a string (supports markdown).
    fn get_info(&self) -> String;
}

/// Contract of a system prompt generator, so an agent can use a custom prompt
/// builder instead of [`SystemPromptGenerator`] (mirrors upstream's
/// `BaseSystemPromptGenerator`).
///
/// ```
/// use atomic_agents::system_prompt::BaseSystemPromptGenerator;
///
/// struct Fixed;
/// impl BaseSystemPromptGenerator for Fixed {
///     fn generate_prompt(&self) -> String {
///         "Be brief.".to_string()
///     }
/// }
/// assert_eq!(Fixed.generate_prompt(), "Be brief.");
/// ```
pub trait BaseSystemPromptGenerator: Send + Sync {
    /// Generates the complete system prompt string.
    fn generate_prompt(&self) -> String;

    /// Registers a dynamic context provider. Generators that do not support
    /// context providers ignore the call.
    fn register_context_provider(&mut self, _name: String, _provider: Box<dyn ContextProvider>) {}

    /// Removes a context provider by name. Ignored by generators that do not
    /// support context providers.
    fn unregister_context_provider(&mut self, _name: &str) {}
}

impl BaseSystemPromptGenerator for SystemPromptGenerator {
    fn generate_prompt(&self) -> String {
        SystemPromptGenerator::generate_prompt(self)
    }

    fn register_context_provider(&mut self, name: String, provider: Box<dyn ContextProvider>) {
        SystemPromptGenerator::register_context_provider(self, name, provider)
    }

    fn unregister_context_provider(&mut self, name: &str) {
        SystemPromptGenerator::unregister_context_provider(self, name)
    }
}

/// Constructs system prompts from structured components.
///
/// Builds prompts with sections for background, steps, output instructions,
/// and dynamic context providers.
///
/// Mirrors Python's `SystemPromptGenerator`.
pub struct SystemPromptGenerator {
    /// Background information about the agent's identity and purpose.
    pub background: Vec<String>,
    /// Internal steps the agent should follow.
    pub steps: Vec<String>,
    /// Instructions for how to format output.
    pub output_instructions: Vec<String>,
    /// Dynamic context providers keyed by name (insertion order preserved).
    pub context_providers: IndexMap<String, Box<dyn ContextProvider>>,
}

impl SystemPromptGenerator {
    /// Creates a new system prompt generator with the given components.
    pub fn new() -> Self {
        Self {
            background: vec![
                "This is a conversation with a helpful and friendly AI assistant.".to_string(),
            ],
            steps: Vec::new(),
            output_instructions: vec![
                "Always respond using the proper JSON schema.".to_string(),
                "Always use the available additional information and context to enhance the response.".to_string(),
            ],
            context_providers: IndexMap::new(),
        }
    }

    /// Sets the background instructions, replacing defaults.
    pub fn with_background(mut self, background: Vec<String>) -> Self {
        self.background = background;
        self
    }

    /// Sets the steps instructions.
    pub fn with_steps(mut self, steps: Vec<String>) -> Self {
        self.steps = steps;
        self
    }

    /// Sets the output instructions, replacing defaults.
    pub fn with_output_instructions(mut self, instructions: Vec<String>) -> Self {
        self.output_instructions = instructions;
        // Always append the standard instructions
        self.output_instructions
            .push("Always respond using the proper JSON schema.".to_string());
        self.output_instructions.push(
            "Always use the available additional information and context to enhance the response."
                .to_string(),
        );
        self
    }

    /// Registers a context provider.
    pub fn register_context_provider(
        &mut self,
        name: impl Into<String>,
        provider: Box<dyn ContextProvider>,
    ) {
        self.context_providers.insert(name.into(), provider);
    }

    /// Removes a context provider by name.
    pub fn unregister_context_provider(&mut self, name: &str) {
        self.context_providers.swap_remove(name);
    }

    /// Generates the complete system prompt string.
    pub fn generate_prompt(&self) -> String {
        let mut sections = Vec::new();

        // Background section
        if !self.background.is_empty() {
            let mut section = String::from("# IDENTITY and PURPOSE\n");
            for item in &self.background {
                section.push_str(&format!("- {item}\n"));
            }
            sections.push(section);
        }

        // Steps section
        if !self.steps.is_empty() {
            let mut section = String::from("# INTERNAL ASSISTANT STEPS\n");
            for item in &self.steps {
                section.push_str(&format!("- {item}\n"));
            }
            sections.push(section);
        }

        // Output instructions section
        if !self.output_instructions.is_empty() {
            let mut section = String::from("# OUTPUT INSTRUCTIONS\n");
            for item in &self.output_instructions {
                section.push_str(&format!("- {item}\n"));
            }
            sections.push(section);
        }

        // Context providers section
        if !self.context_providers.is_empty() {
            let mut section = String::from("# EXTRA INFORMATION AND CONTEXT\n");
            for provider in self.context_providers.values() {
                section.push_str(&format!("## {}\n", provider.title()));
                section.push_str(&provider.get_info());
                section.push('\n');
            }
            sections.push(section);
        }

        sections.join("\n")
    }
}

impl Default for SystemPromptGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SystemPromptGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemPromptGenerator")
            .field("background", &self.background)
            .field("steps", &self.steps)
            .field("output_instructions", &self.output_instructions)
            .field(
                "context_providers",
                &self.context_providers.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestContextProvider {
        title: String,
        info: String,
    }

    impl ContextProvider for TestContextProvider {
        fn title(&self) -> &str {
            &self.title
        }

        fn get_info(&self) -> String {
            self.info.clone()
        }
    }

    #[test]
    fn test_default_prompt() {
        let spg = SystemPromptGenerator::new();
        let prompt = spg.generate_prompt();
        assert!(prompt.contains("IDENTITY and PURPOSE"));
        assert!(prompt.contains("helpful and friendly AI assistant"));
        assert!(prompt.contains("OUTPUT INSTRUCTIONS"));
        assert!(prompt.contains("JSON schema"));
    }

    #[test]
    fn test_custom_background() {
        let spg = SystemPromptGenerator::new()
            .with_background(vec!["You are a code reviewer.".to_string()]);
        let prompt = spg.generate_prompt();
        assert!(prompt.contains("code reviewer"));
        assert!(!prompt.contains("helpful and friendly"));
    }

    #[test]
    fn test_steps() {
        let spg = SystemPromptGenerator::new().with_steps(vec![
            "Analyze the input.".to_string(),
            "Generate a response.".to_string(),
        ]);
        let prompt = spg.generate_prompt();
        assert!(prompt.contains("INTERNAL ASSISTANT STEPS"));
        assert!(prompt.contains("Analyze the input"));
        assert!(prompt.contains("Generate a response"));
    }

    #[test]
    fn test_context_providers() {
        let mut spg = SystemPromptGenerator::new();
        spg.register_context_provider(
            "search",
            Box::new(TestContextProvider {
                title: "Search Results".to_string(),
                info: "Result 1: Rust is great\nResult 2: Serde is fast".to_string(),
            }),
        );

        let prompt = spg.generate_prompt();
        assert!(prompt.contains("EXTRA INFORMATION AND CONTEXT"));
        assert!(prompt.contains("## Search Results"));
        assert!(prompt.contains("Rust is great"));
    }

    #[test]
    fn test_unregister_context_provider() {
        let mut spg = SystemPromptGenerator::new();
        spg.register_context_provider(
            "search",
            Box::new(TestContextProvider {
                title: "Search".to_string(),
                info: "data".to_string(),
            }),
        );
        assert_eq!(spg.context_providers.len(), 1);
        spg.unregister_context_provider("search");
        assert_eq!(spg.context_providers.len(), 0);
    }
}
