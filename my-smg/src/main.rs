use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use my_smg::{
    config::{GatewayConfig, PolicyKind},
    policy::{FirstHealthy, Policy, RoundRobin},
    worker::Worker,
};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    process,
    sync::{Arc, Mutex},
};
use tokio::net::TcpListener;

#[derive(Deserialize, Serialize)]
struct ChatRequest {
    message: String,
}

struct GatewayState {
    config: GatewayConfig,
    client: reqwest::Client,
    workers: Vec<Worker>,
    policy: Mutex<Box<dyn Policy + Send>>,
}

async fn chat(
    State(state): State<Arc<GatewayState>>,
    Json(request): Json<ChatRequest>,
) -> (StatusCode, String) {
    let selected_idex = {
        let mut policy = match state.policy.lock() {
            Ok(gurad) => gurad,
            Err(posioned) => posioned.into_inner(),
        };
        policy.select(&state.workers)
    };

    let Some(index) = selected_idex else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "no healthy worker".to_string(),
        );
    };

    let address = &state.config.workers[index].address;
    let url = format!("{}/chat", address.trim_end_matches('/'));

    let upstream = match state.client.post(url).json(&request).send().await {
        Ok(response) => response,
        Err(error) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {error}"),
            );
        }
    };

    let status = upstream.status();

    match upstream.text().await {
        Ok(body) => (status, body),
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            format!("upstream response failed: {error}"),
        ),
    }
}

async fn health(State(state): State<Arc<GatewayState>>) -> String {
    format!("ready: {} workers", state.config.workers.len())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(config_path) = std::env::args().nth(1) else {
        eprintln!("usage: my-smg <config-path>");
        process::exit(2);
    };

    let config = match GatewayConfig::from_file(Path::new(&config_path)) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("failed to start: {error}");
            process::exit(1);
        }
    };

    let workers: Vec<Worker> = config
        .workers
        .iter()
        .map(|worker| Worker::new(worker.id.clone()))
        .collect();

    let policy: Box<dyn Policy + Send> = match &config.policy {
        PolicyKind::FirstHealthy => Box::new(FirstHealthy::new()),
        PolicyKind::RoundRobin => Box::new(RoundRobin::new()),
    };

    let state = GatewayState {
        config,
        client: reqwest::Client::builder().build()?,
        workers,
        policy: Mutex::new(policy),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/chat", post(chat))
        .with_state(Arc::new(state));

    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    println!("gateway listening on http://127.0.0.1:3000");

    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use my_smg::{config::WorkerConfig, worker::HealthStatus};

    #[tokio::test]
    async fn returns_503_when_no_workers_is_healthy() {
        let config = GatewayConfig {
            workers: vec![WorkerConfig {
                id: "worker-1".to_string(),
                address: "http://127.0.0.1:3001".to_string(),
            }],
            policy: PolicyKind::RoundRobin,
        };

        let mut worker = Worker::new("worker-1".to_string());
        worker.set_status(HealthStatus::Unhealthy);

        let policy: Box<dyn Policy + Send> = Box::new(RoundRobin::new());
        let state = Arc::new(GatewayState {
            config,
            client: reqwest::Client::new(),
            workers: vec![worker],
            policy: Mutex::new(policy),
        });

        let (status, body) = chat(
            State(state),
            Json(ChatRequest {
                message: "hello".to_string(),
            }),
        )
        .await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body, "no healthy worker");
    }
}
