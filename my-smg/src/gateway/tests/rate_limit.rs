use tokio::time::error::Elapsed;

use super::*;

/// 只封装调用与测试超时，断言留在各测试中。
async fn call_chat(state: &Arc<GatewayState>) -> Result<(StatusCode, String), Elapsed> {
    tokio::time::timeout(
        Duration::from_secs(3),
        chat(
            State(Arc::clone(state)),
            Json(ChatRequest {
                message: "rate-limit-test".to_string(),
            }),
        ),
    )
    .await
}

#[tokio::test]
async fn concurrent_requests_share_one_bucket() {
    let (server, request_count) =
        start_retry_server(0, StatusCode::SERVICE_UNAVAILABLE, Duration::ZERO).await;

    let worker = Arc::new(Worker::new(
        "worker-a".to_string(),
        server.address().to_string(),
    ));

    // 使用较长补充间隔，避免测试期间恢复额度。
    let bucket = TokenBucket::new(2, Duration::from_secs(60), Instant::now())
        .expect("valid test bucket configuration");

    let state = gateway_state_with_bucket(&[Arc::clone(&worker)], bucket);

    let (first, second, third) =
        tokio::join!(call_chat(&state), call_chat(&state), call_chat(&state),);

    // 先清理服务器，再检查结果。
    let server_result = server.stop().await;

    let responses = [
        first.expect("first request should finish"),
        second.expect("second request should finish"),
        third.expect("third request should finish"),
    ];

    let accepted = responses
        .iter()
        .filter(|response| response.0 == StatusCode::OK)
        .count();

    let rejected = responses
        .iter()
        .filter(|response| response.0 == StatusCode::TOO_MANY_REQUESTS)
        .count();

    assert_eq!(accepted, 2);
    assert_eq!(rejected, 1);

    for (status, body) in responses {
        if status == StatusCode::TOO_MANY_REQUESTS {
            assert_eq!(body, "rate limit exceeded");
        }
    }

    // 被限流的请求没有联系上游。
    assert_eq!(request_count.load(Ordering::Relaxed), 2);
    assert_eq!(worker.counter(), 0);

    server_result.expect("test server should stop without panicking");
}

#[tokio::test]
async fn retries_consume_only_one_token() {
    // 前两次上游尝试失败，第三次成功。
    let (server, request_count) =
        start_retry_server(2, StatusCode::SERVICE_UNAVAILABLE, Duration::ZERO).await;

    let worker = Arc::new(Worker::new(
        "worker-a".to_string(),
        server.address().to_string(),
    ));

    let bucket = TokenBucket::new(1, Duration::from_secs(60), Instant::now())
        .expect("valid test bucket configuration");

    let state = gateway_state_with_bucket(&[Arc::clone(&worker)], bucket);

    let first = call_chat(&state).await;
    let count_after_first = request_count.load(Ordering::Relaxed);

    let second = call_chat(&state).await;
    let count_after_second = request_count.load(Ordering::Relaxed);

    let server_result = server.stop().await;

    let (first_status, first_body) = first.expect("first request should finish");

    // 只有一枚令牌，但同一次逻辑请求仍能重试至成功。
    assert_eq!(first_status, StatusCode::OK);
    assert_eq!(first_body, "successful response");
    assert_eq!(count_after_first, 3);

    let (second_status, second_body) = second.expect("second request should finish");

    assert_eq!(second_status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(second_body, "rate limit exceeded");

    // 第二次逻辑请求被拒绝，上游次数不再增加。
    assert_eq!(count_after_second, count_after_first);
    assert_eq!(worker.counter(), 0);

    server_result.expect("test server should stop without panicking");
}
