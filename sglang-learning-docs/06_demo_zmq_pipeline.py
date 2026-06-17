"""ZMQ pipeline demo for Week 1.

Run from repo root:
    python sglang-learning-docs/06_demo_zmq_pipeline.py
"""

import multiprocessing as mp
import socket
import threading
import time

try:
    import zmq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyzmq is not installed. Run: bash sglang-learning-docs/setup/setup_mac.sh"
    ) from exc


def tokenizer(addr_in: str, addr_out: str):
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    push = ctx.socket(zmq.PUSH)
    pull.connect(addr_in)
    push.bind(addr_out)

    msg = pull.recv_json()
    text = msg["text"]
    token_ids = [ord(ch) for ch in text]
    print(f"[tokenizer] text={text!r} -> token_ids={token_ids}", flush=True)
    push.send_json({"rid": msg["rid"], "input_ids": token_ids})


def scheduler(addr_in: str, addr_out: str):
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    push = ctx.socket(zmq.PUSH)
    pull.connect(addr_in)
    push.bind(addr_out)

    msg = pull.recv_json()
    output_ids = msg["input_ids"] + [33]
    print(f"[scheduler] input_ids={msg['input_ids']} -> output_ids={output_ids}", flush=True)
    push.send_json({"rid": msg["rid"], "output_ids": output_ids})


def detokenizer(addr_in: str, addr_out: str):
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    push = ctx.socket(zmq.PUSH)
    pull.connect(addr_in)
    push.bind(addr_out)

    msg = pull.recv_json()
    text = "".join(chr(i) for i in msg["output_ids"])
    print(f"[detokenizer] output_ids={msg['output_ids']} -> text={text!r}", flush=True)
    push.send_json({"rid": msg["rid"], "text": text})


def run_tcp_process_pipeline():
    def free_addr() -> str:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return f"tcp://127.0.0.1:{sock.getsockname()[1]}"

    http_to_tok = free_addr()
    tok_to_sched = free_addr()
    sched_to_detok = free_addr()
    detok_to_http = free_addr()

    procs = [
        mp.Process(target=tokenizer, args=(http_to_tok, tok_to_sched)),
        mp.Process(target=scheduler, args=(tok_to_sched, sched_to_detok)),
        mp.Process(target=detokenizer, args=(sched_to_detok, detok_to_http)),
    ]
    for proc in procs:
        proc.start()

    ctx = zmq.Context()
    send = ctx.socket(zmq.PUSH)
    recv = ctx.socket(zmq.PULL)
    send.bind(http_to_tok)
    recv.connect(detok_to_http)

    time.sleep(0.2)
    print("[http] send request", flush=True)
    send.send_json({"rid": "req-1", "text": "Hi"})
    result = recv.recv_json()
    print(f"[http] response={result}", flush=True)

    for proc in procs:
        proc.join(timeout=2)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def run_inproc_fallback():
    print("[fallback] local bind is not permitted; use zmq inproc:// with threads", flush=True)
    ctx = zmq.Context()
    http_to_tok = "inproc://http_to_tok"
    tok_to_sched = "inproc://tok_to_sched"
    sched_to_detok = "inproc://sched_to_detok"
    detok_to_http = "inproc://detok_to_http"

    def tokenizer_thread():
        pull = ctx.socket(zmq.PULL)
        push = ctx.socket(zmq.PUSH)
        pull.connect(http_to_tok)
        push.bind(tok_to_sched)
        msg = pull.recv_json()
        token_ids = [ord(ch) for ch in msg["text"]]
        print(f"[tokenizer] text={msg['text']!r} -> token_ids={token_ids}", flush=True)
        push.send_json({"rid": msg["rid"], "input_ids": token_ids})

    def scheduler_thread():
        pull = ctx.socket(zmq.PULL)
        push = ctx.socket(zmq.PUSH)
        pull.connect(tok_to_sched)
        push.bind(sched_to_detok)
        msg = pull.recv_json()
        output_ids = msg["input_ids"] + [33]
        print(f"[scheduler] input_ids={msg['input_ids']} -> output_ids={output_ids}", flush=True)
        push.send_json({"rid": msg["rid"], "output_ids": output_ids})

    def detokenizer_thread():
        pull = ctx.socket(zmq.PULL)
        push = ctx.socket(zmq.PUSH)
        pull.connect(sched_to_detok)
        push.bind(detok_to_http)
        msg = pull.recv_json()
        text = "".join(chr(i) for i in msg["output_ids"])
        print(f"[detokenizer] output_ids={msg['output_ids']} -> text={text!r}", flush=True)
        push.send_json({"rid": msg["rid"], "text": text})

    threads = [
        threading.Thread(target=tokenizer_thread),
        threading.Thread(target=scheduler_thread),
        threading.Thread(target=detokenizer_thread),
    ]
    for thread in threads:
        thread.start()

    send = ctx.socket(zmq.PUSH)
    recv = ctx.socket(zmq.PULL)
    send.bind(http_to_tok)
    recv.connect(detok_to_http)
    time.sleep(0.1)
    print("[http] send request", flush=True)
    send.send_json({"rid": "req-1", "text": "Hi"})
    print(f"[http] response={recv.recv_json()}", flush=True)

    for thread in threads:
        thread.join(timeout=2)


def main():
    try:
        run_tcp_process_pipeline()
    except (PermissionError, zmq.ZMQError):
        run_inproc_fallback()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
