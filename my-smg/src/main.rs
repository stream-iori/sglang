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

fn is_retryable_status(status: StatusCode) -> bool {
    match status {
        StatusCode::BAD_GATEWAY | StatusCode::GATEWAY_TIMEOUT | StatusCode::SERVICE_UNAVAILABLE => {
            true
        }
        _ => false,
    }
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

    const MAX_ATTEMPTS: usize = 3;
    let mut attempt = 1;
    loop {
        // 每次重新构建请求；借用 URL 和消息，不取走它们。
        let upstream = match state
            .client
            .post(url.as_str())
            .json(&request)
            .timeout(Duration::from_secs(1))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                // 网络错误直接返回，不参与本轮重试。
                return upstream_error_response(&error, "request");
            }
        };

        let status = upstream.status();

        let body = match upstream.text().await {
            Ok(body) => body,
            Err(error) => {
                return upstream_error_response(&error, "response");
            }
        };

        if is_retryable_status(status) && attempt < MAX_ATTEMPTS {
            attempt += 1;
            continue;
        }

        // 成功、不可重试，或次数用尽，都返回本次完整响应。
        return (status, body);
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        task::JoinError,
        time::error::Elapsed,
    };

    // ── 测试资源与准备工具 ──

    /// 持有后台任务：正常路径显式取消并等待，提前退出时由 Drop 请求取消。
    struct ManagedTask<T> {
        handle: Option<JoinHandle<T>>,
    }

    impl<T> ManagedTask<T> {
        fn new(handle: JoinHandle<T>) -> Self {
            Self {
                handle: Some(handle),
            }
        }

        //`mut self` 不是 `&mut self`：方法取得对象所有权，调用后原变量不能再使用
        async fn cancel_and_join(mut self) -> Result<T, JoinError> {
            // take() 把句柄移出并留下 None，避免之后 Drop 再次处理同一个句柄。
            let handle = self
                .handle
                .take()
                .expect("managed task should own its handle");
            handle.abort();
            // abort 只是请求取消；await 返回后，才能确认任务和它的 guard 已结束。
            handle.await
        }
    }

    impl<T> Drop for ManagedTask<T> {
        fn drop(&mut self) {
            if let Some(handle) = &self.handle {
                // Drop 不能异步等待。这是提前退出的兜底，不代替正常路径的清理。
                handle.abort();
            }
        }
    }

    /// 地址与服务器任务一起持有，避免只保存地址而忘记清理后台任务。
    struct TestServer {
        address: String,
        task: ManagedTask<()>,
    }

    impl TestServer {
        fn address(&self) -> &str {
            &self.address
        }

        async fn stop(self) -> Result<(), JoinError> {
            match self.task.cancel_and_join().await {
                Ok(()) => Ok(()),
                // 服务器允许已经完成；取消属于正常清理，但 panic 不能被吞掉。
                Err(error) if error.is_cancelled() => Ok(()),
                Err(error) => Err(error),
            }
        }
    }

    /// 直接准备注册表，不通过被测的 add_worker()，避免准备过程依赖被测行为。
    fn gateway_state_with_workers(workers: &[Arc<Worker>]) -> Arc<GatewayState> {
        let mut registry = WorkerRegistry::new();
        for worker in workers {
            // 克隆 Arc 只增加共享所有权，不复制 Worker；测试仍能观察同一个节点。
            assert!(
                registry.insert(Arc::clone(worker)),
                "fixture worker IDs should be unique"
            );
        }

        Arc::new(GatewayState {
            client: reqwest::Client::new(),
            registry: Mutex::new(registry),
            policy: Mutex::new(Box::new(RoundRobin::new())),
        })
    }

    async fn start_retry_server(
        failures_before_success: usize,
        failure_status: StatusCode,
    ) -> (TestServer, Arc<AtomicUsize>) {
        let request_count = Arc::new(AtomicUsize::new(0));
        let handler_count = Arc::clone(&request_count);

        let app = Router::new().route(
            "/chat",
            post(move || {
                // 每次请求取得自己的 Arc，再移入本次 Future。
                let request_count = Arc::clone(&handler_count);

                async move {
                    // fetch_add 返回增加前的值：第一条请求得到 0。
                    let previous_count = request_count.fetch_add(1, Ordering::Relaxed);

                    if previous_count < failures_before_success {
                        (failure_status, "simulated failure")
                    } else {
                        (StatusCode::OK, "successful response")
                    }
                }
            }),
        );

        let server = start_test_server(app).await;

        (server, request_count)
    }

    fn empty_gateway_state() -> Arc<GatewayState> {
        gateway_state_with_workers(&[])
    }

    /// 使用随机端口，保证多个测试可以并行运行而不争抢固定端口。
    async fn start_test_server(app: Router) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");

        let address = listener
            .local_addr()
            .expect("test server should have an address");

        let task = ManagedTask::new(tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test server should run");
        }));

        TestServer {
            address: format!("http://{address}"),
            task,
        }
    }

    async fn start_health_server(status: StatusCode) -> TestServer {
        let app = Router::new().route("/health", get(move || async move { status }));
        start_test_server(app).await
    }

    async fn start_slow_chat_server(delay: Duration) -> TestServer {
        let app = Router::new().route(
            "/chat",
            post(move || async move {
                tokio::time::sleep(delay).await;
                "slow response"
            }),
        );
        start_test_server(app).await
    }

    /// 分别控制响应头和正文的时间，才能专门覆盖 text().await 的超时分支。
    async fn start_slow_body_server(delay: Duration) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");
        let address = listener
            .local_addr()
            .expect("test server should have an address");
        let task = ManagedTask::new(tokio::spawn(async move {
            let (mut socket, _) = listener
                .accept()
                .await
                .expect("test server should accept a connection");
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
                // TCP 可分多次送达；只累积本次读到的字节，直到请求头结束。
                received.extend_from_slice(&buffer[..count]);
                if received.windows(4).any(|part| part == b"\r\n\r\n") {
                    break;
                }
            }

            // 声明正文还有 5 字节：收到 200 响应头不代表完整响应已收到。
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\n")
                .await
                .expect("test server should send response headers");
            tokio::time::sleep(delay).await;
            // 客户端超时后可能已经断开，因此允许这次正文写入失败。
            let _ = socket.write_all(b"hello").await;
        }));

        TestServer {
            address: format!("http://{address}"),
            task,
        }
    }

    /// 等待可观察状态，不靠固定睡眠猜测任务是否启动；超时结果由测试自行断言。
    async fn wait_until(
        limit: Duration,
        mut condition: impl FnMut() -> bool,
    ) -> Result<(), Elapsed> {
        tokio::time::timeout(limit, async {
            while !condition() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
    }

    // ── 节点管理 ──

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
        let original = Arc::new(Worker::new(
            "worker-a".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));

        let state = gateway_state_with_workers(&[Arc::clone(&original)]);

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
    async fn retries_only_retryable_responses_within_attempt_limit() {
        // 每项：失败次数、失败状态、预期状态、预期请求次数。
        let cases = [
            (2, StatusCode::SERVICE_UNAVAILABLE, StatusCode::OK, 3),
            (
                3,
                StatusCode::SERVICE_UNAVAILABLE,
                StatusCode::SERVICE_UNAVAILABLE,
                3,
            ),
            (3, StatusCode::BAD_REQUEST, StatusCode::BAD_REQUEST, 1),
        ];

        for (failures, failure_status, expected_status, expected_count) in cases {
            let (server, request_count) = start_retry_server(failures, failure_status).await;

            let worker = Arc::new(Worker::new(
                "worker-a".to_string(),
                server.address().to_string(),
            ));

            let state = gateway_state_with_workers(&[Arc::clone(&worker)]);

            let result = tokio::time::timeout(
                Duration::from_secs(3),
                chat(
                    State(state),
                    Json(ChatRequest {
                        message: "retry-check".to_string(),
                    }),
                ),
            )
            .await;

            // 先清理，再断言；测试失败也不会跳过正常清理步骤。
            let server_result = server.stop().await;

            let (status, body) = result.expect("chat should finish within the test deadline");

            assert_eq!(status, expected_status);

            let expected_body = if expected_status == StatusCode::OK {
                "successful response"
            } else {
                "simulated failure"
            };

            assert_eq!(body, expected_body);
            assert_eq!(request_count.load(Ordering::Relaxed), expected_count,);
            assert_eq!(worker.counter(), 0);
            assert_eq!(worker.status(), HealthStatus::Healthy);

            server_result.expect("test server should stop without panicking");
        }
    }

    #[tokio::test]
    async fn removes_worker_once_and_returns_404_afterwards() {
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));

        let state = gateway_state_with_workers(&[worker]);

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
    async fn handlers_reflect_worker_removal() {
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));

        let state = gateway_state_with_workers(&[worker]);

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

    // ── 健康检查 ──

    #[tokio::test]
    async fn health_probe_recovers_worker_after_200() {
        let server = start_health_server(StatusCode::OK).await;

        let worker = Worker::new("worker-a".to_string(), server.address().to_string());
        worker.set_status(HealthStatus::Unhealthy);

        let client = reqwest::Client::new();

        check_worker_health(&client, &worker).await;
        let actual = worker.status();

        server
            .stop()
            .await
            .expect("test server should stop without panicking");
        assert_eq!(actual, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn health_probe_marks_worker_unhealthy_after_503() {
        let server = start_health_server(StatusCode::SERVICE_UNAVAILABLE).await;

        let worker = Worker::new("worker-a".to_string(), server.address().to_string());
        assert_eq!(worker.status(), HealthStatus::Healthy);

        let client = reqwest::Client::new();

        check_worker_health(&client, &worker).await;
        let actual = worker.status();

        server
            .stop()
            .await
            .expect("test server should stop without panicking");
        assert_eq!(actual, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn background_health_checks_recover_worker() {
        let server = start_health_server(StatusCode::OK).await;

        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            server.address().to_string(),
        ));
        worker.set_status(HealthStatus::Unhealthy);

        let state = gateway_state_with_workers(&[Arc::clone(&worker)]);
        let health_task = ManagedTask::new(start_health_checks(
            Arc::clone(&state),
            Duration::from_millis(10),
        ));

        let recovered = wait_until(Duration::from_secs(3), || {
            worker.status() == HealthStatus::Healthy
        })
        .await;

        let health_result = health_task.cancel_and_join().await;
        let server_result = server.stop().await;

        assert!(
            recovered.is_ok(),
            "background checks should recover the worker"
        );

        let health_error = health_result.expect_err("health task should be cancelled");
        assert!(health_error.is_cancelled());

        server_result.expect("test server should stop without panicking");

        assert_eq!(worker.status(), HealthStatus::Healthy);
    }

    // ── 在途计数与任务管理 ──

    #[tokio::test]
    async fn resets_count_when_task_is_cancelled() {
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "127.0.0.1:3212".to_string(),
        ));
        let task_worker = Arc::clone(&worker);

        let task = ManagedTask::new(tokio::spawn(async move {
            let _guard = task_worker.begin_request();

            //pending::<()>() 产生一个永远不会完成的 Future，让任务停在等待状态
            std::future::pending::<()>().await;
        }));

        let started = wait_until(Duration::from_secs(2), || worker.counter() == 1).await;
        let result = task.cancel_and_join().await;
        assert!(started.is_ok(), "task should create its guard in time");

        let error = result.expect_err("task should be cancelled");
        assert!(error.is_cancelled());
        assert_eq!(worker.counter(), 0);
    }

    #[tokio::test]
    async fn counts_two_concurrent_requests_for_same_worker() {
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "127.0.0.1:3212".to_string(),
        ));
        let task_a_worker = Arc::clone(&worker);
        let task_a = ManagedTask::new(tokio::spawn(async move {
            let _guard = task_a_worker.begin_request();
            std::future::pending::<()>().await;
        }));

        let task_b_worker = Arc::clone(&worker);
        let task_b = ManagedTask::new(tokio::spawn(async move {
            let _guard = task_b_worker.begin_request();
            std::future::pending::<()>().await;
        }));

        let started = wait_until(Duration::from_secs(2), || worker.counter() == 2).await;

        let result_a = task_a.cancel_and_join().await;
        let remaining_after_a = worker.counter();

        let result_b = task_b.cancel_and_join().await;

        assert!(started.is_ok(), "both tasks should create their guards");

        let error_a = result_a.expect_err("task A should be cancelled");
        assert!(error_a.is_cancelled());
        assert_eq!(remaining_after_a, 1);

        let error_b = result_b.expect_err("task B should be cancelled");
        assert!(error_b.is_cancelled());
        assert_eq!(worker.counter(), 0);
    }

    #[tokio::test]
    async fn resets_count_when_managed_task_is_dropped() {
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            "127.0.0.1:3212".to_string(),
        ));
        let task_worker = Arc::clone(&worker);
        let task = ManagedTask::new(tokio::spawn(async move {
            let _guard = task_worker.begin_request();
            std::future::pending::<()>().await;
        }));

        let started = wait_until(Duration::from_secs(2), || worker.counter() == 1).await;

        // 模拟测试提前退出：没有显式 join，Drop 仍应请求取消任务。
        drop(task);
        // Drop 不等待取消完成，因此不能马上断言计数为 0。
        let released = wait_until(Duration::from_secs(2), || worker.counter() == 0).await;

        started.expect("task should create its guard");
        released.expect("dropping the managed task should eventually release its guard");
        assert_eq!(worker.counter(), 0);
    }

    // ── HTTP 转发、超时与取消 ──

    #[tokio::test]
    async fn returns_503_when_no_worker_is_healthy() {
        let worker = Arc::new(Worker::new(
            "worker-1".to_string(),
            "http://127.0.0.1:3001".to_string(),
        ));
        worker.set_status(HealthStatus::Unhealthy);

        let state = gateway_state_with_workers(&[worker]);

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
    async fn returns_504_and_resets_count_when_upstream_is_slow() {
        let server = start_slow_chat_server(Duration::from_millis(1500)).await;
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            server.address().to_string(),
        ));
        let state = gateway_state_with_workers(&[Arc::clone(&worker)]);

        // 3 秒是测试兜底；生产 chat 内的 1 秒才是请求超时。
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

        let server_result = server.stop().await;
        let (status, body) = result.expect("chat should finish within the test deadline");

        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(body, "upstream request timed out");
        assert_eq!(worker.counter(), 0);
        assert_eq!(worker.status(), HealthStatus::Healthy);
        server_result.expect("test server should stop without panicking");
    }

    #[tokio::test]
    async fn returns_504_when_upstream_body_is_slow() {
        let server = start_slow_body_server(Duration::from_millis(1500)).await;
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            server.address().to_string(),
        ));
        let state = gateway_state_with_workers(&[Arc::clone(&worker)]);

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

        let server_result = server.stop().await;
        let (status, body) = result.expect("chat should finish within the test deadline");

        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        // response 而非 request：证明超时发生在读取正文时。
        assert_eq!(body, "upstream response timed out");
        assert_eq!(worker.counter(), 0);
        assert_eq!(worker.status(), HealthStatus::Healthy);
        server_result.expect("test server should stop without panicking");
    }

    #[tokio::test]
    async fn resets_count_when_chat_is_cancelled() {
        let server = start_slow_chat_server(Duration::from_secs(60)).await;
        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            server.address().to_string(),
        ));
        let state = gateway_state_with_workers(&[Arc::clone(&worker)]);
        let request_state = Arc::clone(&state);

        let request_task = ManagedTask::new(tokio::spawn(async move {
            chat(
                State(request_state),
                Json(ChatRequest {
                    message: "cancel-check".to_string(),
                }),
            )
            .await
        }));

        // 计数为 1 表示 guard 已创建，不表示假 Worker 已经收到请求。
        let started = wait_until(Duration::from_secs(2), || worker.counter() == 1).await;

        // 必须在断言前清理两个任务；取消 chat 的结果仍由本测试明确验证。
        let request_result = request_task.cancel_and_join().await;
        let server_result = server.stop().await;

        started.expect("chat should become in flight");
        let error = request_result.expect_err("chat task should be cancelled");
        assert!(error.is_cancelled());
        assert_eq!(worker.counter(), 0);
        assert_eq!(worker.status(), HealthStatus::Healthy);
        server_result.expect("test server should stop without panicking");
    }

    #[test]
    fn recognizes_retryable_upstream_statuses() {
        let statuses = [
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ];

        for status in statuses {
            assert!(is_retryable_status(status), "{status} should be retryable");
        }
    }

    #[test]
    fn rejects_non_retryable_upstream_statuses() {
        let statuses = [
            StatusCode::OK,
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::TOO_MANY_REQUESTS,
        ];

        for status in statuses {
            assert!(
                !is_retryable_status(status),
                "{status} should not be retryable under the current policy"
            );
        }
    }
}
