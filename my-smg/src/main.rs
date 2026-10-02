use axum::{
    Json, Router,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    routing::{delete, get, post},
};
use my_smg::{
    config::{GatewayConfig, PolicyKind, WorkerConfig},
    policy::{FirstHealthy, Policy, RoundRobin},
    worker::{HealthStatus, Worker},
    worker_registry::WorkerRegistry,
};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    process,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{net::TcpListener, task::JoinHandle};

#[derive(Deserialize, Serialize)]
struct ChatRequest {
    message: String,
}

struct GatewayState {
    client: reqwest::Client,
    registry: Mutex<WorkerRegistry>,
    policy: Mutex<Box<dyn Policy + Send>>,
}

fn upstream_error_response(error: &reqwest::Error, phase: &str) -> (StatusCode, String) {
    if error.is_timeout() {
        (
            StatusCode::GATEWAY_TIMEOUT,
            format!("upstream {phase} timed out"),
        )
    } else {
        (
            StatusCode::BAD_GATEWAY,
            format!("upstream {phase} failed: {error}"),
        )
    }
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
    //意思是这行下面的return都会减一
    let _inflight_guard = worker.begin_request();
    let url = format!("{}/chat", worker.address().trim_end_matches('/'));

    let upstream = match state
        .client
        .post(url)
        .json(&request)
        .timeout(Duration::from_secs(1))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            return upstream_error_response(&error, "request");
        }
    };

    let status = upstream.status();

    match upstream.text().await {
        Ok(body) => (status, body),
        Err(error) => upstream_error_response(&error, "response"),
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
        output.push_str(&format!(
            "{}: inflight={} health={}\n",
            worker.id(),
            worker.counter(),
            worker.status(),
        ));
    }

    output
}

async fn add_worker(
    State(state): State<Arc<GatewayState>>,
    Json(request): Json<WorkerConfig>,
) -> (StatusCode, String) {
    // 临时配置只用于复用校验，不改变网关策略。
    let config = GatewayConfig {
        workers: vec![request],
        policy: PolicyKind::FirstHealthy,
    };

    if let Err(error) = config.validate() {
        return (StatusCode::BAD_REQUEST, format!("invalid worker: {error}"));
    }

    let worker_config = &config.workers[0];

    let worker = Arc::new(Worker::new(
        worker_config.id.clone(),
        worker_config.address.clone(),
    ));

    let inserted = {
        let mut registry = match state.registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        registry.insert(worker)
    };

    if inserted {
        (StatusCode::CREATED, "worker registered".to_string())
    } else {
        (StatusCode::CONFLICT, "worker id already exists".to_string())
    }
}

async fn remove_worker(
    State(state): State<Arc<GatewayState>>,
    AxumPath(worker_id): AxumPath<String>,
) -> StatusCode {
    let removed = {
        let mut registry = match state.registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        registry.remove(&worker_id)
    };

    if removed {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn check_worker_health(client: &reqwest::Client, worker: &Worker) {
    let url = format!("{}/health", worker.address().trim_end_matches('/'));

    let healthy = match client
        .get(url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    };

    let status = if healthy {
        HealthStatus::Healthy
    } else {
        HealthStatus::Unhealthy
    };

    worker.set_status(status);
}

fn start_health_checks(state: Arc<GatewayState>, pause: Duration) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let snapshot = {
                let registry = match state.registry.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                registry.snapshot()
            };

            for worker in &snapshot {
                check_worker_health(&state.client, worker).await;
            }

            tokio::time::sleep(pause).await;
        }
    })
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

    let state = Arc::new(GatewayState {
        client: reqwest::Client::builder().build()?,
        registry: Mutex::new(registry),
        policy: Mutex::new(policy),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/chat", post(chat))
        .route("/workers", post(add_worker))
        .route("/workers/{worker_id}", delete(remove_worker))
        .with_state(state.clone());

    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    println!("gateway listening on http://127.0.0.1:3000");

    let health_task = start_health_checks(state.clone(), Duration::from_secs(1));

    let serve_result = axum::serve(listener, app).await;

    //这里的意思是服务结束,比如Ctrl+c, 会触发到下面的代码，停止后台任务
    health_task.abort();

    if let Err(error) = health_task.await {
        if !error.is_cancelled() {
            eprintln!("health task failed:{error}");
        }
    }

    serve_result?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use my_smg::worker::HealthStatus;

    async fn start_health_server(status: StatusCode) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");

        let address = listener
            .local_addr()
            .expect("test server should have an address");

        let app = Router::new().route("/health", get(move || async move { status }));

        let server = tokio::spawn(async {
            axum::serve(listener, app)
                .await
                .expect("test server should run");
        });

        (format!("http://{address}"), server)
    }

    fn empty_gateway_state() -> Arc<GatewayState> {
        let policy = Box::new(RoundRobin::new());

        Arc::new(GatewayState {
            client: reqwest::Client::new(),
            registry: Mutex::new(WorkerRegistry::new()),
            policy: Mutex::new(policy),
        })
    }

    #[tokio::test]
    async fn health_probe_recovers_worker_after_200() {
        let (address, server) = start_health_server(StatusCode::OK).await;

        let worker = Worker::new("worker-a".to_string(), address);
        worker.set_status(HealthStatus::Unhealthy);

        let client = reqwest::Client::new();

        check_worker_health(&client, &worker).await;
        let actual = worker.status();

        server.abort();
        let error = server.await.expect_err("test server should be cancelled");

        assert!(error.is_cancelled());
        assert_eq!(actual, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn health_probe_marks_worker_unhealthy_after_503() {
        let (address, server) = start_health_server(StatusCode::SERVICE_UNAVAILABLE).await;

        let worker = Worker::new("worker-a".to_string(), address);
        assert_eq!(worker.status(), HealthStatus::Healthy);

        let client = reqwest::Client::new();

        check_worker_health(&client, &worker).await;
        let actual = worker.status();

        server.abort();
        let error = server.await.expect_err("test server should be cancelled");

        assert!(error.is_cancelled());
        assert_eq!(actual, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn registers_valid_worker() {
        let state = empty_gateway_state();

        let (status, body) = add_worker(
            State(Arc::clone(&state)),
            Json(WorkerConfig {
                id: "worker-a".to_string(),
                address: "http://127.0.0.1:3001".to_string(),
            }),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(body, "worker registered");

        let worker = {
            let registry = state
                .registry
                .lock()
                .expect("registry lock should not be poisoned");

            registry
                .get("worker-a")
                .expect("registered worker should exist")
        };

        assert_eq!(worker.address(), "http://127.0.0.1:3001");
    }

    #[tokio::test]
    async fn duplicate_registration_preserves_original_worker() {
        let state = empty_gateway_state();

        let original = Arc::new(Worker::new(
            "worker-a".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));

        {
            let mut registry = state
                .registry
                .lock()
                .expect("registry lock should not be poisoned");

            assert!(registry.insert(Arc::clone(&original)));
        }

        let (status, body) = add_worker(
            State(Arc::clone(&state)),
            Json(WorkerConfig {
                id: "worker-a".to_string(),
                address: "http://127.0.0.1:3002".to_string(),
            }),
        )
        .await;

        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body, "worker id already exists");

        let found = {
            let registry = state
                .registry
                .lock()
                .expect("registry lock should not be poisoned");

            assert_eq!(registry.snapshot().len(), 1);

            registry
                .get("worker-a")
                .expect("original worker should remain")
        };

        assert!(Arc::ptr_eq(&original, &found));
        assert_eq!(found.address(), "http://127.0.0.1:3001");
    }

    #[tokio::test]
    async fn rejects_invalid_address_without_registering_worker() {
        let state = empty_gateway_state();

        let (status, body) = add_worker(
            State(Arc::clone(&state)),
            Json(WorkerConfig {
                id: "worker-a".to_string(),
                address: "not-a-url".to_string(),
            }),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.starts_with("invalid worker:"));

        let registry = state
            .registry
            .lock()
            .expect("registry lock should not be poisoned");

        assert!(registry.snapshot().is_empty());
    }

    #[tokio::test]
    async fn removes_worker_once_and_returns_404_afterwards() {
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));

        let mut registry = WorkerRegistry::new();
        assert!(registry.insert(worker));

        let policy = Box::new(RoundRobin::new());

        let state = Arc::new(GatewayState {
            client: reqwest::Client::new(),
            registry: Mutex::new(registry),
            policy: Mutex::new(policy),
        });

        let first =
            remove_worker(State(Arc::clone(&state)), AxumPath("worker-a".to_string())).await;

        assert_eq!(first, StatusCode::NO_CONTENT);

        let output = health(State(Arc::clone(&state))).await;
        assert_eq!(output, "ready: 0 workers\n");

        let second =
            remove_worker(State(Arc::clone(&state)), AxumPath("worker-a".to_string())).await;

        assert_eq!(second, StatusCode::NOT_FOUND);
    }

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

    #[tokio::test]
    async fn background_health_checks_recover_worker() {
        let (address, server) = start_health_server(StatusCode::OK).await;

        let worker = Arc::new(Worker::new("worker-a".to_string(), address));
        worker.set_status(HealthStatus::Unhealthy);

        let state = empty_gateway_state();

        {
            let mut registry = state
                .registry
                .lock()
                .expect("registry lock should not be poisoned");

            assert!(registry.insert(Arc::clone(&worker)));
        }

        let health_task = start_health_checks(Arc::clone(&state), Duration::from_millis(10));

        let recovered = tokio::time::timeout(Duration::from_secs(3), async {
            while worker.status() != HealthStatus::Healthy {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await;

        health_task.abort();
        let health_result = health_task.await;

        server.abort();
        let server_result = server.await;

        assert!(
            recovered.is_ok(),
            "background checks should recover the worker"
        );

        let health_error = health_result.expect_err("health task should be cancelled");
        assert!(health_error.is_cancelled());

        let server_error = server_result.expect_err("test server should be cancelled");
        assert!(server_error.is_cancelled());

        assert_eq!(worker.status(), HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn handlers_reflect_worker_removal() {
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));

        let mut registry = WorkerRegistry::new();
        assert!(registry.insert(worker));

        let policy: Box<dyn Policy + Send> = Box::new(RoundRobin::new());

        let state = Arc::new(GatewayState {
            client: reqwest::Client::new(),
            registry: Mutex::new(registry),
            policy: Mutex::new(policy),
        });

        let before = health(State(Arc::clone(&state))).await;

        assert_eq!(
            before,
            "ready: 1 workers\nworker-a: inflight=0 health=Healthy\n"
        );

        {
            let mut registry = state
                .registry
                .lock()
                .expect("registry lock should not be poisoned");

            assert!(registry.remove("worker-a"));
        }

        let after = health(State(Arc::clone(&state))).await;

        assert_eq!(after, "ready: 0 workers\n");

        let (status, body) = chat(
            State(Arc::clone(&state)),
            Json(ChatRequest {
                message: "hello".to_string(),
            }),
        )
        .await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body, "no healthy worker");
    }

    #[tokio::test]
    async fn returns_504_and_resets_count_when_upstream_is_slow() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");

        let address = listener
            .local_addr()
            .expect("test server should have an address");

        let app = Router::new().route(
            "/chat",
            post(|| async {
                tokio::time::sleep(Duration::from_millis(1500)).await;
                "slow response"
            }),
        );

        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test server should run");
        });

        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            format!("http://{address}"),
        ));

        let state = empty_gateway_state();

        {
            let mut registry = state
                .registry
                .lock()
                .expect("registry lock should not be poisoned");

            assert!(registry.insert(Arc::clone(&worker)));
        }

        let result = tokio::time::timeout(
            Duration::from_secs(3),
            chat(
                State(Arc::clone(&state)),
                Json(ChatRequest {
                    message: "timeout-check".to_string(),
                }),
            ),
        )
        .await;

        server.abort();
        let server_result = server.await;

        let (status, body) = result.expect("chat should finish within the test deadline");

        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(body, "upstream request timed out");
        assert_eq!(worker.counter(), 0);
        assert_eq!(worker.status(), HealthStatus::Healthy);

        let error = server_result.expect_err("test server should be cancelled");
        assert!(error.is_cancelled());
    }

    #[tokio::test]
    async fn returns_504_when_upstream_body_is_slow() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");

        let address = listener
            .local_addr()
            .expect("test server should have an address");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener
                .accept()
                .await
                .expect("test server should accept a connection");

            // TCP 可能分多次送达，读到请求头结束为止。
            let mut received = Vec::new();
            let mut buffer = [0u8; 1024];

            loop {
                let count = socket
                    .read(&mut buffer)
                    .await
                    .expect("test server should read the request");

                assert!(
                    count > 0,
                    "connection closed before request headers arrived"
                );

                //把本次读到的字节，追加到 received，保留之前收到的内容, [..count] 取下标0到count,不包含count
                received.extend_from_slice(&buffer[..count]);

                if received.windows(4).any(|part| part == b"\r\n\r\n") {
                    break;
                }
            }

            // 先发响应头，声明正文还有 5 字节。
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\n")
                .await
                .expect("test server should send response headers");

            // 正文延迟超过 chat 中的 1 秒请求超时。
            tokio::time::sleep(Duration::from_millis(1500)).await;

            // 客户端可能已经超时断开，因此允许写入失败。
            let _ = socket.write_all(b"hello").await;
        });

        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            format!("http://{address}"),
        ));

        let state = empty_gateway_state();

        {
            let mut registry = state
                .registry
                .lock()
                .expect("registry lock should not be poisoned");

            assert!(registry.insert(Arc::clone(&worker)));
        }

        // 3 秒只是测试兜底，不是网关的请求超时。
        let result = tokio::time::timeout(
            Duration::from_secs(3),
            chat(
                State(Arc::clone(&state)),
                Json(ChatRequest {
                    message: "body-timeout-check".to_string(),
                }),
            ),
        )
        .await;

        //发出取消请求
        server.abort();

        //等待结束,并拿到结束的结果
        let server_result = server.await;

        let (status, body) = result.expect("chat should finish within the test deadline");

        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(body, "upstream response timed out");
        assert_eq!(worker.counter(), 0);
        assert_eq!(worker.status(), HealthStatus::Healthy);

        // 清理任务：允许任务已结束，也允许被 abort 取消。
        if let Err(error) = server_result {
            assert!(error.is_cancelled(), "test server failed: {error}");
        }
    }
}
