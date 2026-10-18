//! 网关运行逻辑：共享状态、HTTP 处理器和后台健康检查。
//! 模块仍属于 binary；不向 library 公开教学内部接口。
mod retry;

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::StatusCode,
};
use my_smg::{
    circuit_breaker::{self, gate::CircuitGateError},
    config::{GatewayConfig, PolicyKind, WorkerConfig},
    policy::Policy,
    worker::{HealthStatus, Worker},
    worker_registry::WorkerRegistry,
};
use retry::retry_with_backoff;
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::task::JoinHandle;

#[derive(Deserialize, Serialize)]
pub(super) struct ChatRequest {
    message: String,
}

// 仅向父模块开放，供 main 组装状态；不是对外的公共库 API。
// 测试是本模块的子模块，因此也能访问这里的内部实现。
pub(super) struct GatewayState {
    pub(super) client: reqwest::Client,
    pub(super) registry: Mutex<WorkerRegistry>,
    pub(super) policy: Mutex<Box<dyn Policy + Send>>,
}

fn is_retryable_status(status: StatusCode) -> bool {
    match status {
        StatusCode::BAD_GATEWAY | StatusCode::GATEWAY_TIMEOUT | StatusCode::SERVICE_UNAVAILABLE => {
            true
        }
        _ => false,
    }
}

fn circuit_gate_error_response(error: CircuitGateError) -> (StatusCode, String) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        format!("circuit gate unavailable: {error}"),
    )
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

pub(super) async fn chat(
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

    // 一个 guard 对应整次逻辑请求，不是每次重试。
    // 它覆盖请求、读响应体和退避；正常返回、总超时或任务取消时都会 Drop。
    let _inflight_guard = worker.begin_request();
    let url = format!("{}/chat", worker.address().trim_end_matches('/'));

    // 每次操作只复制这些共享引用，不移动原来的对象。
    let client = &state.client;
    let url_ref = url.as_str();
    let request_ref = &request;
    let gate = worker.circuit_gate();

    // 闭包可以重复调用，每次创建一个独立的请求 Future
    // 每次调用闭包，都为本次上游尝试申请一个独立凭证
    let operation = move || async move {
        let permit = match gate.acquire(Instant::now()) {
            Ok(Some(permit)) => permit,
            Ok(None) => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    "worker circuit rejected request".to_string(),
                ));
            }

            Err(error) => {
                return Err(circuit_gate_error_response(error));
            }
        };

        let upstream = match client
            .post(url_ref)
            .json(request_ref)
            .timeout(Duration::from_secs(1))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                //发送失败，单次请求超时，都计入节点失败.
                //若Gate 更新失败，优先返回Gate内部错误。
                permit
                    .failure(Instant::now())
                    .map_err(circuit_gate_error_response)?;
                return Err(upstream_error_response(&error, "request"));
            }
        };

        let status = upstream.status();

        let body = match upstream.text().await {
            Ok(body) => body,
            Err(error) => {
                permit
                    .failure(Instant::now())
                    .map_err(circuit_gate_error_response)?;

                return Err(upstream_error_response(&error, "response"));
            }
        };

        // 必须等完整正文读完，再报告本次尝试的结果。
        // 熔断判断与重试判断分开：所有5xx都计入熔断失败。
        if status.is_server_error() {
            permit
                .failure(Instant::now())
                .map_err(circuit_gate_error_response)?;
        } else {
            // 非5xx不归咎于节点故障，不表示业务请求一定成功。
            permit
                .success(Instant::now())
                .map_err(circuit_gate_error_response)?;
        }

        Ok((status, body))
    };

    // 包住完整重试过程，包括 HTTP 请求、读正文和退避。
    let result = tokio::time::timeout(Duration::from_secs(2), retry_with_backoff(operation)).await;

    match result {
        // 收到了完整上游响应。
        Ok(Ok(response)) => response,

        // 发送或读取失败：返回已经映射好的错误响应。
        Ok(Err(response)) => response,

        // 整个重试过程超时。
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            "upstream total deadline exceeded".to_string(),
        ),
    }
}

pub(super) async fn health(State(state): State<Arc<GatewayState>>) -> String {
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

pub(super) async fn add_worker(
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

pub(super) async fn remove_worker(
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

pub(super) fn start_health_checks(state: Arc<GatewayState>, pause: Duration) -> JoinHandle<()> {
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

// 只在测试构建中加载 src/gateway/tests.rs，正式 binary 不包含测试工具。
#[cfg(test)]
mod tests;
