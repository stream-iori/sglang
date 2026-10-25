//! 资源生命周期：多个任务共享同一 Worker，各自的 guard 独立释放。
//! 取消必须等任务结束；Drop 只请求取消时，要等待计数实际归零。

use super::*;
use futures_util::StreamExt;
use rstest::rstest;

fn stream_worker_state(server: &TestServer) -> (Arc<Worker>, Arc<GatewayState>) {
    let worker = Arc::new(Worker::new(
        "worker-a".to_string(),
        server.address().to_string(),
    ));

    let state = gateway_state_with_workers(&[Arc::clone(&worker)]);

    (worker, state)
}

async fn stream_fixture() -> (TestServer, Arc<Worker>, Arc<GatewayState>) {
    let app = Router::new().route(
        "/chat/stream",
        post(|Json(_request): Json<ChatRequest>| async {
            ([(CONTENT_TYPE, "text/event-stream")], "data: hello\n\n")
        }),
    );

    let server = start_test_server(app).await;
    let (worker, state) = stream_worker_state(&server);

    (server, worker, state)
}

async fn request_stream(state: Arc<GatewayState>) -> Response {
    tokio::time::timeout(
        Duration::from_secs(2),
        chat_stream(
            State(state),
            Json(ChatRequest {
                message: "stream-request".to_string(),
            }),
        ),
    )
    .await
    .expect("stream handler should return within the test deadline")
}

#[tokio::test]
async fn stream_count_lasts_until_body_reaches_end() {
    let (server, worker, state) = stream_fixture().await;

    let response = request_stream(Arc::clone(&state)).await;

    let status = response.status();
    let count_after_handler = worker.counter();
    let permits_after_handler = state.concurrency.available_permits();

    let mut body = response.into_body().into_data_stream();

    let collected = tokio::time::timeout(Duration::from_secs(2), async {
        let mut collected = Vec::new();

        while let Some(chunk) = body.next().await {
            let chunk = chunk.expect("body chunk should be readable");
            collected.extend_from_slice(&chunk);
        }

        collected
    })
    .await;

    // 在销毁 body 前记录，验证读取结束本身就能释放 guard 和执行名额。
    let count_after_end = worker.counter();
    let permits_after_end = state.concurrency.available_permits();

    drop(body);
    let server_result = server.stop().await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(count_after_handler, 1);
    assert_eq!(permits_after_handler, 1);

    let collected = collected.expect("body should finish in time");
    assert_eq!(collected.as_slice(), b"data: hello\n\n");

    assert_eq!(count_after_end, 0);
    assert_eq!(permits_after_end, 2);
    server_result.expect("test server should stop");
}

#[tokio::test]
async fn resets_count_when_task_is_cancelled() {
    let worker = idle_worker();
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
    let worker = idle_worker();
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
    let worker = idle_worker();
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

#[rstest]
#[case::unread(false)]
#[case::after_first_chunk(true)]
#[tokio::test]
async fn dropping_stream_body_releases_count(#[case] read_first_chunk: bool) {
    let (server, worker, state) = stream_fixture().await;

    let response = request_stream(Arc::clone(&state)).await;

    let status = response.status();
    let count_after_handler = worker.counter();
    let permits_after_handler = state.concurrency.available_permits();

    let mut body = response.into_body().into_data_stream();

    if read_first_chunk {
        let first = tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .expect("first chunk should arrive in time")
            .expect("body should contain a chunk")
            .expect("first chunk should be readable");

        assert!(!first.is_empty());
    }

    // 没有继续拉取到 None，流仍持有 guard。
    let count_before_drop = worker.counter();
    let permits_before_drop = state.concurrency.available_permits();

    // 销毁正文流，连带销毁内部状态和 guard。
    drop(body);

    let count_after_drop = worker.counter();
    let permits_after_drop = state.concurrency.available_permits();

    // 正常路径先清理服务器，再检查结果。
    let server_result = server.stop().await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(count_after_handler, 1);
    assert_eq!(permits_after_handler, 1);
    assert_eq!(count_before_drop, 1);
    assert_eq!(permits_before_drop, 1);
    assert_eq!(count_after_drop, 0);
    assert_eq!(permits_after_drop, 2);

    server_result.expect("test server should stop");
}

#[tokio::test]
async fn stream_send_failure_releases_count() {
    let (server, worker, state) = stream_fixture().await;

    // 先停止服务器，使后续连接失败。
    server.stop().await.expect("test server should stop");

    assert_eq!(worker.counter(), 0);
    assert_eq!(state.concurrency.available_permits(), 2);

    let response = request_stream(Arc::clone(&state)).await;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

    // 发送失败时，handler 提前返回，局部 guard 和执行名额应已释放。
    assert_eq!(worker.counter(), 0);
    assert_eq!(state.concurrency.available_permits(), 2);
}

#[tokio::test]
async fn stream_bodies_hold_execution_permits() {
    let (server, worker, state) = stream_fixture().await;

    // 两个 handler 都返回，但正文还没有读取或销毁。
    let response_a = request_stream(Arc::clone(&state)).await;
    let response_b = request_stream(Arc::clone(&state)).await;

    let status_a = response_a.status();
    let status_b = response_b.status();

    let permits_when_full = state.concurrency.available_permits();
    let inflight_when_full = worker.counter();

    // 没有执行名额，等待容量也为 0。
    let response_c = request_stream(Arc::clone(&state)).await;

    let status_c = response_c.status();
    let inflight_after_rejection = worker.counter();

    // 销毁一个响应正文，释放一个执行名额。
    drop(response_a);

    let permits_after_a = state.concurrency.available_permits();
    let inflight_after_a = worker.counter();

    // 归还的名额应该可以被新请求使用。
    let response_d = request_stream(Arc::clone(&state)).await;

    let status_d = response_d.status();
    let permits_after_d = state.concurrency.available_permits();
    let inflight_after_d = worker.counter();

    drop(response_b);
    drop(response_d);
    drop(response_c);

    let permits_after_cleanup = state.concurrency.available_permits();
    let inflight_after_cleanup = worker.counter();

    let server_result = server.stop().await;

    assert_eq!(status_a, StatusCode::OK);
    assert_eq!(status_b, StatusCode::OK);
    assert_eq!(permits_when_full, 0);
    assert_eq!(inflight_when_full, 2);

    assert_eq!(status_c, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(inflight_after_rejection, 2);

    assert_eq!(permits_after_a, 1);
    assert_eq!(inflight_after_a, 1);

    assert_eq!(status_d, StatusCode::OK);
    assert_eq!(permits_after_d, 0);
    assert_eq!(inflight_after_d, 2);

    assert_eq!(permits_after_cleanup, 2);
    assert_eq!(inflight_after_cleanup, 0);

    server_result.expect("test server should stop");
}

//已取得响应头，但读取正文失败，释放计数
#[tokio::test]
async fn stream_read_failure_releases_count() {
    // 服务器先发送响应头，再等待正文。
    let server = start_slow_body_server(Duration::from_secs(60)).await;

    let (worker, state) = stream_worker_state(&server);

    let response = request_stream(Arc::clone(&state)).await;

    let status = response.status();
    let count_after_handler = worker.counter();
    let permits_after_handler = state.concurrency.available_permits();

    let mut body = response.into_body().into_data_stream();

    // 响应声明还有 5 字节正文，但服务器尚未发送。
    // 此时停止服务器，制造正文不完整的读取错误。
    server.stop().await.expect("test server should stop");

    let next = tokio::time::timeout(Duration::from_secs(2), body.next()).await;

    // 必须在 drop(body) 之前记录。
    // 验证读取错误本身已经让流释放 guard 和执行名额。
    let count_after_error = worker.counter();
    let permits_after_error = state.concurrency.available_permits();

    drop(body);

    assert_eq!(status, StatusCode::OK);
    assert_eq!(count_after_handler, 1);
    assert_eq!(permits_after_handler, 1);

    let item = next
        .expect("body read should finish in time")
        .expect("body should report a read error");

    assert!(item.is_err());
    assert_eq!(count_after_error, 0);
    assert_eq!(permits_after_error, 2);
}
