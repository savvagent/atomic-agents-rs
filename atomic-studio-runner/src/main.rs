use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use atomic_studio_runner::pipeline::PipelineDefinition;
use atomic_studio_runner::server;
use atomic_studio_runner::validate::validate_pipeline;

#[derive(Parser)]
#[command(name = "atomic-studio-runner")]
#[command(about = "Run and validate atomic-agents pipeline definitions")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a pipeline definition file.
    Validate {
        /// Path to the pipeline JSON file.
        path: PathBuf,
    },
    /// Run a pipeline with the given input.
    Run {
        /// Path to the pipeline JSON file.
        path: PathBuf,
        /// JSON input string.
        #[arg(long)]
        input: String,
    },
    /// Start the HTTP server that powers the atomic-studio PWA.
    Serve {
        /// Address to bind to.
        #[arg(long, default_value = "127.0.0.1:8787")]
        addr: SocketAddr,
        /// Optional directory containing the built PWA (e.g. `atomic-studio/build`).
        /// When set, the contents are served as the fallback route.
        #[arg(long)]
        static_dir: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init()
        .ok();

    let cli = Cli::parse();

    match cli.command {
        Commands::Validate { path } => {
            let pipeline = PipelineDefinition::from_file(&path)?;
            let report = validate_pipeline(&pipeline);

            if report.errors.is_empty() && report.warnings.is_empty() {
                println!("Pipeline is valid.");
            } else {
                for error in &report.errors {
                    eprintln!(
                        "ERROR{}: {}",
                        error
                            .node_id
                            .as_ref()
                            .map(|id| format!(" [node: {id}]"))
                            .unwrap_or_default(),
                        error.message
                    );
                }
                for warning in &report.warnings {
                    eprintln!(
                        "WARNING{}: {}",
                        warning
                            .node_id
                            .as_ref()
                            .map(|id| format!(" [node: {id}]"))
                            .unwrap_or_default(),
                        warning.message
                    );
                }
                if !report.is_valid() {
                    std::process::exit(1);
                }
            }
        }
        Commands::Run { path, input } => {
            let _pipeline = PipelineDefinition::from_file(&path)?;
            let _input: serde_json::Value = serde_json::from_str(&input)?;

            // Full execution requires a ClientFactory implementation.
            // For now, print the parsed pipeline info.
            println!("Pipeline loaded: {} nodes, {} edges", _pipeline.nodes.len(), _pipeline.edges.len());
            println!("Input: {}", serde_json::to_string_pretty(&_input)?);
            println!("Run command requires a ClientFactory implementation (use as library for full execution).");
        }
        Commands::Serve { addr, static_dir } => {
            server::serve(addr, static_dir).await?;
        }
    }

    Ok(())
}
