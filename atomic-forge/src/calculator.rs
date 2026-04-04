use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Tool for performing calculations. Supports basic arithmetic operations
/// like addition, subtraction, multiplication, and division, as well as
/// exponentiation, modulo, and parenthesized expressions.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct CalculatorInput {
    /// Mathematical expression to evaluate. For example, '2 + 2' or '(3 + 4) * 2'.
    pub expression: String,
}

/// Result of a calculation.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct CalculatorOutput {
    /// Result of the calculation as a string.
    pub result: String,
}

/// Configuration for the Calculator tool.
pub struct CalculatorConfig {
    pub component: ComponentConfig,
}

impl Default for CalculatorConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("Calculator")
                .with_description("Evaluates mathematical expressions"),
        }
    }
}

/// Tool for evaluating mathematical expressions.
///
/// Uses the `evalexpr` crate for safe expression evaluation.
pub struct CalculatorTool {
    config: ComponentConfig,
}

impl CalculatorTool {
    pub fn new(config: CalculatorConfig) -> Self {
        Self {
            config: config.component,
        }
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new(CalculatorConfig::default())
    }
}

impl Tool for CalculatorTool {
    type Input = CalculatorInput;
    type Output = CalculatorOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let result = evalexpr::eval(&input.expression)
            .map_err(|e| Error::Validation(format!("Failed to evaluate expression: {e}")))?;

        Ok(CalculatorOutput {
            result: result.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let calc = CalculatorTool::default();
        let result = calc
            .run(CalculatorInput {
                expression: "2 + 2".to_string(),
            })
            .unwrap();
        assert_eq!(result.result, "4");
    }

    #[test]
    fn test_complex_expression() {
        let calc = CalculatorTool::default();
        let result = calc
            .run(CalculatorInput {
                expression: "(3 + 4) * 2".to_string(),
            })
            .unwrap();
        assert_eq!(result.result, "14");
    }

    #[test]
    fn test_division() {
        let calc = CalculatorTool::default();
        let result = calc
            .run(CalculatorInput {
                expression: "10.0 / 3.0".to_string(),
            })
            .unwrap();
        // evalexpr uses float division with float operands
        assert!(result.result.starts_with("3.3333"));
    }

    #[test]
    fn test_invalid_expression() {
        let calc = CalculatorTool::default();
        let result = calc.run(CalculatorInput {
            expression: "invalid".to_string(),
        });
        assert!(result.is_err());
    }
}
