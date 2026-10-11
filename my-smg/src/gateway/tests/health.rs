//! 健康检查：探测失败标记不健康，后续成功探测恢复健康。

use super::*;

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
