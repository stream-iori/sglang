#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fake_manifest="$script_dir/../Cargo.toml"
gateway_dir=$(cd "$script_dir/../../.." && pwd)
logs_dir=$(mktemp -d)
pids=()

cleanup() {
  local status=$?
  stop_processes
  if (( status == 0 )); then
    rm -rf "$logs_dir"
  else
    echo "Smoke check failed; logs preserved at $logs_dir" >&2
  fi
  return "$status"
}
trap cleanup EXIT

stop_processes() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  pids=()
}

wait_http() {
  local url=$1
  for _ in $(seq 1 100); do
    if curl --silent --fail "$url" >/dev/null; then return 0; fi
    sleep 0.1
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

cargo build --quiet --manifest-path "$fake_manifest"
cargo build --quiet --manifest-path "$gateway_dir/Cargo.toml" --bin smg
fake_bin="$script_dir/../target/debug/sgl-model-gateway-fake-worker"
gateway_bin="$gateway_dir/target/debug/smg"

"$fake_bin" --port 18001 --model-id regular-fake >"$logs_dir/regular.log" 2>&1 &
pids+=("$!")
wait_http http://127.0.0.1:18001/health
"$gateway_bin" launch --worker-urls http://127.0.0.1:18001 --policy round_robin --host 127.0.0.1 --port 18000 >"$logs_dir/regular-gateway.log" 2>&1 &
pids+=("$!")
wait_http http://127.0.0.1:18000/health
curl --silent --fail http://127.0.0.1:18000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"regular-fake","messages":[{"role":"user","content":"hello"}]}' >/dev/null
curl --silent --fail --no-buffer http://127.0.0.1:18000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"regular-fake","messages":[{"role":"user","content":"hello"}],"stream":true}' | rg -q '\[DONE\]'

stop_processes

"$fake_bin" --port 18011 --model-id pd-fake --worker-type prefill >"$logs_dir/prefill.log" 2>&1 &
pids+=("$!")
"$fake_bin" --port 18012 --model-id pd-fake --worker-type decode >"$logs_dir/decode.log" 2>&1 &
pids+=("$!")
wait_http http://127.0.0.1:18011/health
wait_http http://127.0.0.1:18012/health
"$gateway_bin" launch --pd-disaggregation --prefill http://127.0.0.1:18011 19011 --decode http://127.0.0.1:18012 --policy round_robin --host 127.0.0.1 --port 18010 >"$logs_dir/pd-gateway.log" 2>&1 &
pids+=("$!")
wait_http http://127.0.0.1:18010/health
curl --silent --fail http://127.0.0.1:18010/generate -H 'content-type: application/json' -d '{"text":"pd smoke","stream":false}' >/dev/null
curl --silent http://127.0.0.1:18011/__fake__/state | rg -q 'bootstrap_room'

echo "Fake worker regular and PD smoke checks passed."
