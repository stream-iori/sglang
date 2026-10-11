//! 转发行为：可重试状态、退避、单次/总超时，以及取消后停止请求。
//! 外围 timeout 是测试保险，chat 内部的 timeout 才是被测行为。

use super::*;

#[tokio::test]
async fn waits_retry_before_upstream() {
    let cases = [(2, StatusCode::SERVICE_UNAVAILABLE, StatusCode::OK, 3)];
    for (failures, failure_status, expected_status, expected_count) in cases {
        let (server, request_count) =
            start_retry_server(failures, failure_status, Duration::ZERO).await;

        let worker = Arc::new(Worker::new(
            "worker-a".to_string(),
            server.address().to_string(),
        ));

        let state = gateway_state_with_workers(&[Arc::clone(&worker)]);

        let started = std::time::Instant::now();

        let result = tokio::time::timeout(
            Duration::from_secs(3),
            chat(
                State(state),
                Json(ChatRequest {
                    message: "retry-request".to_string(),
                }),
            ),
        )
        .await;

        let elapsed = started.elapsed();

        // 先清理，再断言，避免断言失败跳过清理。
        let server_result = server.stop().await;

        let (status, body) = result.expect("chat should finish within the test deadline");

        assert_eq!(status, expected_status, "response body: {body}");
        assert_eq!(body, "successful response");
        assert_eq!(request_count.load(Ordering::Relaxed), expected_count,);
        assert!(
            elapsed >= Duration::from_millis(300),
            "expected at least 300ms of backoff, actual: {elapsed:?}",
        );
        assert_eq!(worker.counter(), 0);

        server_result.expect("test server should stop without panicking");
    }
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
        let (server, request_count) =
            start_retry_server(failures, failure_status, Duration::ZERO).await;

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

        assert_eq!(
            status,
            expected_status,
            "response body: {body}; worker received {} requests",
            request_count.load(Ordering::Relaxed),
        );

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
async fn returns_504_when_retry_total_deadline_expires() {
    // 每次响应不到 1 秒，但三次请求加退避会超过总期限。
    let (server, _request_count) = start_retry_server(
        2,
        StatusCode::SERVICE_UNAVAILABLE,
        Duration::from_millis(750),
    )
    .await;

    let worker = Arc::new(Worker::new(
        "worker-a".to_string(),
        server.address().to_string(),
    ));
    let state = gateway_state_with_workers(&[Arc::clone(&worker)]);

    // 4 秒是测试保险；被测的是 chat 内部的 2 秒总期限。
    let result = tokio::time::timeout(
        Duration::from_secs(4),
        chat(
            State(state),
            Json(ChatRequest {
                message: "total-deadline-check".to_string(),
            }),
        ),
    )
    .await;

    // 先清理，再断言。
    let server_result = server.stop().await;

    let (status, body) = result.expect("chat should finish within the test deadline");

    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(body, "upstream total deadline exceeded");
    assert_eq!(worker.counter(), 0);

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
async fn stops_retrying_when_chat_is_cancelled() {
    // 持续返回可重试的 503，保留请求计数供测试观察。
    let (server, request_count) =
        start_retry_server(10, StatusCode::SERVICE_UNAVAILABLE, Duration::ZERO).await;
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
                message: "cancel-retry-check".to_string(),
            }),
        )
        .await
    }));

    // 等待假 Worker 真正收到请求，而不是只观察在途计数。
    let started = wait_until(Duration::from_secs(2), || {
        request_count.load(Ordering::Relaxed) >= 1
    })
    .await;

    // 等待任务结束后，才能确认 guard 已经释放。
    let request_result = request_task.cancel_and_join().await;
    let count_after_cancel = request_count.load(Ordering::Relaxed);

    // 保持服务器运行，观察是否还有后续请求。
    tokio::time::sleep(Duration::from_millis(400)).await;
    let count_after_observation = request_count.load(Ordering::Relaxed);

    //assert after clean
    let server_result = server.stop().await;

    started.expect("worker should receive the first request");
    let error = request_result.expect_err("chat task should be cancelled");
    assert!(error.is_cancelled());
    assert_eq!(
        count_after_observation, count_after_cancel,
        "request count should not increase after cancellation",
    );
    assert_eq!(worker.counter(), 0);

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
