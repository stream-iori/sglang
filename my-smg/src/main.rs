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
    worker_registry::WorkerRegistry,
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
    client: reqwest::Client,
    registry: Mutex<WorkerRegistry>,
    policy: Mutex<Box<dyn Policy + Send>>,
}

async fn chat(
    State(state): State<Arc<GatewayState>>,
    Json(request): Json<ChatRequest>,
) -> (StatusCode, String) {
    let snapshot = {
        let registry = match state.registry.lock() {
            Ok(guard) => guard,
            Err(posioned) => posioned.into_inner(),
        };
        registry.snapshot()
    };

    let selected_idex = {
        let mut policy = match state.policy.lock() {
            Ok(gurad) => gurad,
            Err(posioned) => posioned.into_inner(),
        };
        policy.select(&snapshot)
    };

    let Some(index) = selected_idex else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "no healthy worker".to_string(),
        );
    };

    let worker = &snapshot[index];

    //这一行要放在 chat 的外层作用域，不要包进额外的 {}。这样 guard 会保留到完整响应读取结束；
    //遇到提前 return 也会自动减一, 意思是这行上面的return和下面的return都会减一
    let _inflight_guard = worker.begin_request();
    let url = format!("{}/chat", worker.address().trim_end_matches('/'));

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
    let snapshot = {
        let registry = match state.registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        registry.snapshot()
    };

    let mut output = format!("ready: {} workers\n", snapshot.len());

    for worker in &snapshot {
        output.push_str(&format!("{}: inflight={}\n", worker.id(), worker.counter(),));
    }

    output
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

    let mut registry = WorkerRegistry::new();
    for worker_config in &config.workers {
        let worker = Arc::new(Worker::new(
            worker_config.id.clone(),
            worker_config.address.clone(),
        ));

        if !registry.insert(worker) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("duplicate worker id: {}", worker_config.id),
            )
            .into());
        }
    }

    let policy: Box<dyn Policy + Send> = match &config.policy {
        PolicyKind::FirstHealthy => Box::new(FirstHealthy::new()),
        PolicyKind::RoundRobin => Box::new(RoundRobin::new()),
    };

    let state = GatewayState {
        client: reqwest::Client::builder().build()?,
        registry: Mutex::new(registry),
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
    use my_smg::worker::HealthStatus;

    #[tokio::test]
    async fn resets_count_when_task_is_cancelled() {
        use std::sync::Arc;
        use std::time::Duration;

        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "127.0.0.1:3212".to_string(),
        ));
        let task_worker = Arc::clone(&worker);

        let task = tokio::spawn(async move {
            let _guard = task_worker.begin_request();

            //pending::<()>() 产生一个永远不会完成的 Future，让任务停在等待状态
            std::future::pending::<()>().await;
        });

        let started = tokio::time::timeout(Duration::from_secs(2), async {
            while worker.counter() == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await;

        task.abort();

        let result = task.await;
        assert!(started.is_ok(), "task should create its gurad in time");

        let error = result.expect_err("task should be cancelled");
        assert!(error.is_cancelled());
        assert_eq!(worker.counter(), 0);
    }

    #[tokio::test]
    async fn returns_503_when_no_workers_is_healthy() {
        let worker = Arc::new(Worker::new(
            "worker-1".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));
        worker.set_status(HealthStatus::Unhealthy);

        let mut registry = WorkerRegistry::new();
        assert!(registry.insert(worker));

        let policy: Box<dyn Policy + Send> = Box::new(RoundRobin::new());

        let state = Arc::new(GatewayState {
            client: reqwest::Client::new(),
            registry: Mutex::new(registry),
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

    #[tokio::test]
    async fn counts_two_concurrent_requests_for_same_workers() {
        use std::time::Duration;

        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "127.0.0.1:3212".to_string(),
        ));
        let worker_a = Arc::clone(&worker);
        let task_a = tokio::spawn(async move {
            let _guard = worker_a.begin_request();
            std::future::pending::<()>().await;
        });

        let worker_b = Arc::clone(&worker);
        let task_b = tokio::spawn(async move {
            let _guard = worker_b.begin_request();
            std::future::pending::<()>().await;
        });

        let started = tokio::time::timeout(Duration::from_secs(2), async {
            while worker.counter() != 2 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await;

        task_a.abort();
        let result_a = task_a.await;
        let remaining_after_a = worker.counter();

        task_b.abort();
        let result_b = task_b.await;

        assert!(started.is_ok(), "both tasks should create their guards");

        let error_a = result_a.expect_err("task A shoud be cancelled");
        assert!(error_a.is_cancelled());
        assert_eq!(remaining_after_a, 1);

        let error_b = result_b.expect_err("task B should be cancelled");
        assert!(error_b.is_cancelled());
        assert_eq!(worker.counter(), 0);
    }
}
