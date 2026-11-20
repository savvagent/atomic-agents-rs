//! HTTP server for atomic-studio. Serves the PWA build (when present) and
//! exposes a small JSON API for validating and running pipelines.
//!
//! - `POST /api/validate` → returns a [`ValidationResponse`].
//! - `POST /api/runs`     → streams [`ExecutionEvent`]s as newline-delimited JSON
//!                          (NDJSON) over chunked HTTP transfer. Clients consume
//!                          via `fetch().body.getReader()` — no SSE.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::post;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::client_factory::EnvClientFactory;
use crate::interpreter::{ExecutionEvent, PipelineInterpreter};
use crate::pipeline::PipelineDefinition;
use crate::validate::{ValidationError, ValidationWarning, validate_pipeline};

#[derive(Clone, Default)]
struct AppState;

/// Wire-format validation response with an explicit `valid` flag.
#[derive(Serialize)]
struct ValidationResponse {
    valid: bool,
    errors: Vec<ValidationError>,
    warnings: Vec<ValidationWarning>,
}

#[derive(Deserialize)]
struct RunRequest {
    pipeline: PipelineDefinition,
    input: Value,
}

/// Run the HTTP server until the process is terminated.
pub async fn serve(addr: SocketAddr, static_dir: Option<PathBuf>) -> anyhow::Result<()> {
    let state = AppState;

    let api = Router::new()
        .route("/validate", post(validate_handler))
        .route("/runs", post(run_handler));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let mut app = Router::new()
        .nest("/api", api)
        .with_state(state)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    if let Some(dir) = static_dir {
        if dir.exists() {
            let index = dir.join("index.html");
            let serve_dir = ServeDir::new(&dir).not_found_service(ServeFile::new(index));
            app = app.fallback_service(serve_dir);
            tracing::info!("serving PWA from {}", dir.display());
        } else {
            tracing::warn!(
                "static directory {} does not exist; PWA assets will not be served",
                dir.display()
            );
        }
    }

    tracing::info!("atomic-studio-runner listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn validate_handler(
    State(_state): State<AppState>,
    Json(pipeline): Json<PipelineDefinition>,
) -> Json<ValidationResponse> {
    let report = validate_pipeline(&pipeline);
    Json(ValidationResponse {
        valid: report.is_valid(),
        errors: report.errors,
        warnings: report.warnings,
    })
}

async fn run_handler(
    State(_state): State<AppState>,
    Json(req): Json<RunRequest>,
) -> Result<Response, ApiError> {
    let factory = Box::new(EnvClientFactory);
    let interpreter = Arc::new(PipelineInterpreter::new(req.pipeline, factory));
    let mut events = interpreter.subscribe();

    let (tx, rx) = mpsc::channel::<Result<String, Infallible>>(64);

    // Forward broadcast events to the response stream until the channel closes.
    let forward_tx = tx.clone();
    tokio::spawn(async move {
        while let Ok(event) = events.recv().await {
            let line = match serde_json::to_string(&event) {
                Ok(s) => s,
                Err(err) => {
                    let _ = forward_tx
                        .send(Ok(format!(
                            "{{\"event_type\":\"pipeline_failed\",\"error\":\"serialize: {err}\"}}\n"
                        )))
                        .await;
                    break;
                }
            };
            if forward_tx.send(Ok(format!("{line}\n"))).await.is_err() {
                break;
            }
            if matches!(
                event,
                ExecutionEvent::PipelineCompleted { .. } | ExecutionEvent::PipelineFailed { .. }
            ) {
                break;
            }
        }
    });

    // Run the pipeline. Drop the local sender so the response closes once the
    // forwarder finishes.
    let interp = interpreter.clone();
    let input = req.input;
    tokio::spawn(async move {
        if let Err(err) = interp.run(input).await {
            // Ignore: a pipeline_failed event will already have been emitted in
            // most cases. We log it for the operator's benefit.
            tracing::error!("pipeline run error: {err}");
        }
        drop(tx);
    });

    let stream = ReceiverStream::new(rx).map(|chunk| chunk.map(axum::body::Bytes::from));
    let body = Body::from_stream(stream);

    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .map_err(|e| ApiError(e.to_string()))?;
    Ok(response)
}

struct ApiError(String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0).into_response()
    }
}
