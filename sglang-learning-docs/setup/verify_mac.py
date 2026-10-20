"""验证已启动的 Mac 服务，保存响应、环境来源与可选状态日志；不启动或停止服务。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", default="/tmp/sglang-mac-validation")
    parser.add_argument("--status-dir", type=Path, help="服务写入 scheduler.status 的目录；指定后必须验收日志")
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_id = "mac-audit-" + uuid.uuid4().hex[:12]

    def save(name, value):
        (out / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )

    def session():
        client = requests.Session()
        client.trust_env = False
        return client

    def post(path, payload):
        # 每个请求独立 session，避免在线程之间共享 requests.Session。
        with session() as client:
            response = client.post(base + path, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()

    metadata = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "source_worktree_changes": git(
            "diff", "--name-only", "HEAD", "--", "python/sglang/srt", "python/sglang/kernels"
        ).splitlines(),
        "source_worktree_diff_sha256": hashlib.sha256(
            subprocess.check_output(
                ["git", "diff", "--binary", "HEAD", "--", "python/sglang/srt", "python/sglang/kernels"], cwd=ROOT
            )
        ).hexdigest(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "url": base,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "mlx", "requests")
        },
    }
    for name in ("opentelemetry-sdk", "grpcio-tools"):
        try:
            metadata["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    save("environment", metadata)

    with session() as client:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            try:
                health = client.get(base + "/health", timeout=5)
                if health.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(f"服务健康检查在 {args.startup_timeout:g} 秒内未通过")
            time.sleep(1)
        response = client.get(base + "/server_info", timeout=10)
        response.raise_for_status()
        info = response.json()
        save("server-info", info)
    expected = {
        "device": "mps", "attention_backend": "torch_native",
        "sampling_backend": "pytorch", "disable_overlap_schedule": True,
    }
    for key, value in expected.items():
        assert info.get(key) == value, (key, info.get(key), value)
    config = Path(info["model_path"]) / "config.json"
    if config.is_file():
        metadata["model_config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    save("environment", metadata)

    prompt = run_id + ": " + "MPS cache verification: science and geography. " * 24 + "The capital of France is"
    sampling = {"temperature": 0, "ignore_eos": True, "max_new_tokens": 8}
    payload = {"text": prompt, "sampling_params": sampling}
    cold = post("/generate", {**payload, "rid": run_id + "-cold"})
    save("generate-cold", cold)
    warm = post("/generate", {**payload, "rid": run_id + "-warm"})
    save("generate-warm", warm)
    for result in (cold, warm):
        assert result["meta_info"]["completion_tokens"] == 8, result
        assert result["output_ids"] and result["text"], result
    assert cold["output_ids"] == warm["output_ids"], (cold, warm)
    assert warm["meta_info"]["cached_tokens"] > 0, warm

    chat_payload = {
        "model": info["model_path"],
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
        "temperature": 0, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    chat = post("/v1/chat/completions", chat_payload)
    save("chat", chat)
    assert chat["choices"][0]["message"]["content"], chat
    chunks, finished = [], False
    with session() as client:
        with client.post(base + "/v1/chat/completions", json={**chat_payload, "stream": True}, stream=True, timeout=120) as response:
            response.raise_for_status()
            response.encoding = "utf-8"
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    finished = True
                    break
                chunks.append(json.loads(line[6:]))
    save("chat-stream", chunks)
    assert finished, "SSE 缺少 [DONE]"
    stream_text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
    assert stream_text, chunks
    assert stream_text == chat["choices"][0]["message"]["content"], (stream_text, chat)

    # 与串行参考比较；HTTP 并发不等于已经证明设备同批。
    prompts = [
        "A red apple and a blue sky. Continue: The apple is",
        "A green tree and a yellow flower. Continue: The flower is",
    ]
    serial = [post("/generate", {"text": p, "sampling_params": sampling, "rid": f"{run_id}-serial-{i}"}) for i, p in enumerate(prompts)]
    save("isolation-serial", serial)
    def concurrent_request(item):
        i, text = item
        return post("/generate", {"text": text, "sampling_params": sampling, "rid": f"{run_id}-parallel-{i}"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel = list(executor.map(concurrent_request, enumerate(prompts)))
    save("isolation-parallel", parallel)
    for reference, result in zip(serial, parallel):
        assert reference["output_ids"] == result["output_ids"], (reference, result)
        assert result["meta_info"]["completion_tokens"] == 8, result
    followup = post("/generate", {"text": "After both requests, the server can still", "sampling_params": sampling, "rid": run_id + "-followup"})
    save("followup", followup)
    assert followup["meta_info"]["completion_tokens"] == 8, followup

    status_summary = None
    if args.status_dir:
        events = []
        for path in sorted(args.status_dir.glob("*.log*")):
            for line in path.read_text().splitlines():
                start = line.find("{")
                if start < 0:
                    continue
                try:
                    event = json.loads(line[start:])
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "scheduler.status" and any(
                    rid.startswith(run_id) for rid in event.get("running_rids", [])
                ):
                    events.append(event)
        save("scheduler-status", events)
        assert events, "未找到本次 run_id 的 scheduler.status；检查 --enable-metrics 与日志开关"
        assert any(e.get("cur_batch") and e["cur_batch"]["rids"] for e in events), "cur_batch 没有本轮请求"
        assert any(e.get("req_to_token_pool", {}).get("active_rows") for e in events), "缺少请求行映射"
        assert any(e.get("radix_cache", {}).get("class") == "UnifiedRadixCache" for e in events), "实际 cache 不是预期的 UnifiedRadixCache"
        status_summary = {
            "events": len(events),
            "modes": sorted({e["running_batch"]["forward_mode"] for e in events}),
            "multi_request_batch_observed": any(e["running_batch"]["size"] >= 2 for e in events),
        }
    summary = {
        "run_id": run_id, "git_head": metadata["git_head"],
        "url": base, "health": health.status_code, "effective_config": expected,
        "cold_cached_tokens": cold["meta_info"]["cached_tokens"],
        "warm_cached_tokens": warm["meta_info"]["cached_tokens"],
        "completion_tokens": warm["meta_info"]["completion_tokens"],
        "deterministic_output_equal": True,
        "generate_text": warm["text"], "chat_text": chat["choices"][0]["message"]["content"],
        "stream_text": stream_text, "stream_done": finished,
        "concurrent_matches_serial": True, "followup_ok": True,
        "scheduler_status": status_summary,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    save("summary", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"原始响应已保存：{out}")


if __name__ == "__main__":
    main()
