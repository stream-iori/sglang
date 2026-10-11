//! 资源生命周期：多个任务共享同一 Worker，各自的 guard 独立释放。
//! 取消必须等任务结束；Drop 只请求取消时，要等待计数实际归零。

use super::*;

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
