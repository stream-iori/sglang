//! A local, protocol-level SGLang worker for exercising SGL Model Gateway.
//!
//! It intentionally returns deterministic fake inference responses. It does
//! not load a model, allocate KV cache, or implement a bootstrap server.

use std::{
    net::SocketAddr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Router,
};
use clap::Parser;
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(about = "Local fake SGLang worker for SGL Model Gateway development")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8001)]
    port: u16,
    #[arg(long, default_value = "fake-model")]
    model_id: String,
    #[arg(long, default_value = "regular", value_parser = ["regular", "prefill", "decode"])]
    worker_type: String,
    #[arg(long, default_value = "healthy", value_parser = ["healthy", "degraded", "unhealthy"])]
    health: String,
    #[arg(long, default_value_t = 0)]
    delay_ms: u64,
    #[arg(long)]
    failure_status: Option<u16>,
    #[arg(long, default_value_t = 3)]
    stream_chunks: usize,
    #[arg(long, default_value_t = 0)]
    stream_chunk_delay_ms: u64,
    #[arg(long)]
    stream_error_after: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Health {
    Healthy,
    Degraded,
    Unhealthy,
}

impl Health {
    fn is_available(&self) -> bool {
        !matches!(self, Self::Unhealthy)
    }
}

#[derive(Clone, Debug, Serialize)]
struct FakeWorkerConfig {
    model_id: String,
    worker_type: String,
    health: Health,
    delay_ms: u64,
    failure_status: Option<u16>,
    stream_chunks: usize,
    stream_chunk_delay_ms: u64,
    stream_error_after: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ConfigPatch {
    health: Option<Health>,
    delay_ms: Option<u64>,
    failure_status: Option<u16>,
    failure_enabled: Option<bool>,
    stream_chunks: Option<usize>,
    stream_chunk_delay_ms: Option<u64>,
    stream_error_after: Option<usize>,
    stream_error_enabled: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
struct RequestSummary {
    method: String,
    path: String,
    body: Value,
}

#[derive(Clone, Debug, Default, Serialize)]
struct FakeWorkerStats {
    request_count: u64,
    last_request: Option<RequestSummary>,
}

#[derive(Clone)]
struct FakeWorkerState {
    config: Arc<RwLock<FakeWorkerConfig>>,
    stats: Arc<RwLock<FakeWorkerStats>>,
}

impl FakeWorkerState {
    async fn record_request(&self, path: &str, body: &Value) {
        let mut stats = self.stats.write().await;
        stats.request_count += 1;
        stats.last_request = Some(RequestSummary {
            method: "POST".to_string(),
            path: path.to_string(),
            body: body.clone(),
        });
    }
}

#[derive(Serialize)]
struct StateResponse {
    config: FakeWorkerConfig,
    stats: FakeWorkerStats,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch")
        .as_secs()
}

fn error_response(status: u16, message: &str) -> Response {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        Json(json!({
            "error": {"message": message, "type": "fake_worker_error", "code": status.as_u16()}
        })),
    )
        .into_response()
}

async fn apply_delay_and_failure(state: &FakeWorkerState) -> Option<Response> {
    let config = state.config.read().await.clone();
    if config.delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(config.delay_ms)).await;
    }
    config
        .failure_status
        .map(|status| error_response(status, "configured fake worker failure"))
}

async fn health(State(state): State<FakeWorkerState>) -> Response {
    let config = state.config.read().await.clone();
    if !config.health.is_available() {
        return error_response(503, "fake worker is unhealthy");
    }
    Json(json!({"status": config.health, "worker_type": config.worker_type})).into_response()
}

async fn health_generate(State(state): State<FakeWorkerState>) -> Response {
    let config = state.config.read().await.clone();
    if !config.health.is_available() {
        return error_response(503, "fake generation service is unavailable");
    }
    Json(json!({"status": "ok", "queue_length": 0, "processing_time_ms": config.delay_ms}))
        .into_response()
}

async fn server_info(State(state): State<FakeWorkerState>) -> Response {
    let config = state.config.read().await.clone();
    Json(json!({
        // This worker simulates routing protocol only; it does not have local
        // model weights or a tokenizer file to register with SMG.
        "served_model_name": config.model_id,
        "tp_size": 1,
        "dp_size": 1,
        "context_length": 32768,
        "max_num_batched_tokens": 32768,
        "max_prefill_tokens": 16384,
        "disable_radix_cache": false,
        "version": "fake-worker-0.1.0"
    }))
    .into_response()
}

async fn model_info(State(state): State<FakeWorkerState>) -> Response {
    let config = state.config.read().await.clone();
    Json(json!({"served_model_name": config.model_id, "is_generation": true})).into_response()
}

async fn models(State(state): State<FakeWorkerState>) -> Response {
    let config = state.config.read().await.clone();
    Json(json!({
        "object": "list",
        "data": [{"id": config.model_id, "object": "model", "created": now_secs(), "owned_by": "fake-worker"}]
    }))
    .into_response()
}

async fn generate(State(state): State<FakeWorkerState>, Json(body): Json<Value>) -> Response {
    state.record_request("/generate", &body).await;
    if let Some(response) = apply_delay_and_failure(&state).await {
        return response;
    }
    let stream_requested = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if stream_requested {
        return streaming_response(state, "generate").await;
    }
    Json(json!({
        "text": "fake generated response",
        "meta_info": {"prompt_tokens": 3, "completion_tokens": 3, "finish_reason": {"type": "stop", "reason": "length"}}
    }))
    .into_response()
}

async fn chat_completions(
    State(state): State<FakeWorkerState>,
    Json(body): Json<Value>,
) -> Response {
    state.record_request("/v1/chat/completions", &body).await;
    if let Some(response) = apply_delay_and_failure(&state).await {
        return response;
    }
    let stream_requested = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if stream_requested {
        return streaming_response(state, "chat").await;
    }
    let model_id = state.config.read().await.model_id.clone();
    Json(json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4()),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "fake chat response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
    }))
    .into_response()
}

async fn streaming_response(state: FakeWorkerState, endpoint: &'static str) -> Response {
    let config = state.config.read().await.clone();
    let model_id = config.model_id.clone();
    let chunks = config.stream_chunks;
    let delay = std::time::Duration::from_millis(config.stream_chunk_delay_ms);
    let error_after = config.stream_error_after;
    let stream = stream::unfold(0usize, move |index| {
        let model_id = model_id.clone();
        async move {
            if index > chunks {
                return None;
            }
            if error_after == Some(index) {
                return Some((
                    Err(std::io::Error::other("configured fake stream failure")),
                    chunks + 1,
                ));
            }
            if index == chunks {
                return Some((Ok(Event::default().data("[DONE]")), index + 1));
            }
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let data = if endpoint == "chat" {
                json!({
                    "id": format!("chatcmpl-{}", Uuid::new_v4()),
                    "object": "chat.completion.chunk",
                    "created": now_secs(),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": format!("fake-{} ", index)}, "finish_reason": null}]
                })
            } else {
                json!({"text": format!("fake-{} ", index), "meta_info": {"completion_tokens": index + 1}})
            };
            Some((Ok(Event::default().data(data.to_string())), index + 1))
        }
    });
    Sse::new(stream).into_response()
}

async fn state_handler(State(state): State<FakeWorkerState>) -> Json<StateResponse> {
    Json(StateResponse {
        config: state.config.read().await.clone(),
        stats: state.stats.read().await.clone(),
    })
}

async fn update_config(
    State(state): State<FakeWorkerState>,
    Json(patch): Json<ConfigPatch>,
) -> Json<StateResponse> {
    {
        let mut config = state.config.write().await;
        if let Some(health) = patch.health {
            config.health = health;
        }
        if let Some(delay_ms) = patch.delay_ms {
            config.delay_ms = delay_ms;
        }
        if let Some(status) = patch.failure_status {
            config.failure_status = Some(status);
        }
        if patch.failure_enabled == Some(false) {
            config.failure_status = None;
        }
        if let Some(chunks) = patch.stream_chunks {
            config.stream_chunks = chunks;
        }
        if let Some(delay_ms) = patch.stream_chunk_delay_ms {
            config.stream_chunk_delay_ms = delay_ms;
        }
        if let Some(after) = patch.stream_error_after {
            config.stream_error_after = Some(after);
        }
        if patch.stream_error_enabled == Some(false) {
            config.stream_error_after = None;
        }
    }
    state_handler(State(state)).await
}

fn app(state: FakeWorkerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/server_info", get(server_info))
        .route("/model_info", get(model_info))
        .route("/v1/models", get(models))
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/__fake__/state", get(state_handler))
        .route("/__fake__/config", post(update_config))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let state = FakeWorkerState {
        config: Arc::new(RwLock::new(FakeWorkerConfig {
            model_id: args.model_id,
            worker_type: args.worker_type,
            health: match args.health.as_str() {
                "healthy" => Health::Healthy,
                "degraded" => Health::Degraded,
                "unhealthy" => Health::Unhealthy,
                _ => unreachable!("clap validates health"),
            },
            delay_ms: args.delay_ms,
            failure_status: args.failure_status,
            stream_chunks: args.stream_chunks,
            stream_chunk_delay_ms: args.stream_chunk_delay_ms,
            stream_error_after: args.stream_error_after,
        })),
        stats: Arc::new(RwLock::new(FakeWorkerStats::default())),
    };
    let address: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    println!("Fake worker listening on http://{}", address);
    axum::serve(tokio::net::TcpListener::bind(address).await?, app(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn test_state() -> FakeWorkerState {
        FakeWorkerState {
            config: Arc::new(RwLock::new(FakeWorkerConfig {
                model_id: "test-model".to_string(),
                worker_type: "regular".to_string(),
                health: Health::Healthy,
                delay_ms: 0,
                failure_status: None,
                stream_chunks: 2,
                stream_chunk_delay_ms: 0,
                stream_error_after: None,
            })),
            stats: Arc::new(RwLock::new(FakeWorkerStats::default())),
        }
    }

    #[tokio::test]
    async fn discovery_and_chat_follow_sglang_http_shapes() {
        let router = app(test_state());
        let models = router
            .clone()
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(models.status(), StatusCode::OK);
        let model_body = models.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<Value>(&model_body).unwrap()["data"][0]["id"],
            "test-model"
        );

        let chat = router
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat.status(), StatusCode::OK);
        let chat_body = chat.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<Value>(&chat_body).unwrap()["choices"][0]["message"]
                ["content"],
            "fake chat response"
        );
    }

    #[tokio::test]
    async fn control_api_changes_health_and_failure_behavior() {
        let router = app(test_state());
        let update = router
            .clone()
            .oneshot(
                Request::post("/__fake__/config")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"health":"unhealthy","failure_status":503}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(update.status(), StatusCode::OK);

        let health = router
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);

        let generate = router
            .oneshot(
                Request::post("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(generate.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
