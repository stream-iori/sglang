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

    let response = request_stream(state).await;

    let status = response.status();
    let count_after_handler = worker.counter();

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

    // 在销毁 body 前记录，验证读取结束本身就能释放 guard。
    let count_after_end = worker.counter();

    drop(body);
    let server_result = server.stop().await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(count_after_handler, 1);

    let collected = collected.expect("body should finish in time");
    assert_eq!(collected.as_slice(), b"data: hello\n\n");

    assert_eq!(count_after_end, 0);
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

    let response = request_stream(state).await;

    let status = response.status();
    let count_after_handler = worker.counter();

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

    // 销毁正文流，连带销毁内部状态和 guard。
    drop(body);

    let count_after_drop = worker.counter();

    // 正常路径先清理服务器，再检查结果。
    let server_result = server.stop().await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(count_after_handler, 1);
    assert_eq!(count_before_drop, 1);
    assert_eq!(count_after_drop, 0);

    server_result.expect("test server should stop");
}
