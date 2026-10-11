//! 程序入口：读取配置、组装网关状态和路由、启动服务。
//! 请求处理与测试放在 gateway 模块，本文件不展开转发细节。

mod gateway;

use axum::{
    Router,
    routing::{delete, get, post},
};
use gateway::{GatewayState, add_worker, chat, health, remove_worker, start_health_checks};
use my_smg::{
    config::{GatewayConfig, PolicyKind},
    policy::{FirstHealthy, Policy, RoundRobin},
    worker::Worker,
    worker_registry::WorkerRegistry,
};
use std::{
    path::Path,
    process,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::net::TcpListener;

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

    // serve 返回后，停止并等待后台健康检查任务。
    // 默认 Ctrl+C 会终止进程，不保证执行到这里；优雅停机需另行接入关闭信号。
    health_task.abort();

    if let Err(error) = health_task.await {
        if !error.is_cancelled() {
            eprintln!("health task failed:{error}");
        }
    }

    serve_result?;

    Ok(())
}
