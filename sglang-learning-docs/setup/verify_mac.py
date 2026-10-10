"""验证已启动的 mac 服务；保存原始响应，不自动启动或停止服务。"""
import argparse
import json
import time
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:30000')
parser.add_argument('--output', default='/tmp/sglang-mac-validation')
args = parser.parse_args()
base = args.url.rstrip('/')
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.trust_env = False  # 本机 loopback 不经过代理。


def save(name, value):
    (out / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))


def post(path, payload):
    response = session.post(base + path, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# 启动期间可能是 503；等待预热完成，最多 120 秒。
deadline = time.monotonic() + 120
while True:
    try:
        health = session.get(base + '/health', timeout=5)
        if health.status_code == 200:
            break
    except requests.RequestException:
        pass
    if time.monotonic() >= deadline:
        raise RuntimeError('服务健康检查在 120 秒内未通过')
    time.sleep(1)
info = session.get(base + '/server_info', timeout=10)
info.raise_for_status()
save('server-info', info.json())

prompt = 'MPS cache verification: science and geography. ' * 24 + 'The capital of France is'
payload = {'text': prompt, 'sampling_params': {'temperature': 0, 'ignore_eos': True, 'max_new_tokens': 8}}
cold = post('/generate', payload)
warm = post('/generate', payload)
save('generate-cold', cold)
save('generate-warm', warm)
assert cold['meta_info']['completion_tokens'] == 8, cold
assert warm['meta_info']['completion_tokens'] == 8, warm
assert cold['output_ids'] == warm['output_ids'], (cold, warm)
assert warm['meta_info']['cached_tokens'] > 0, warm
assert isinstance(warm['text'], str) and warm['text'], warm

chat_payload = {
    'model': info.json()['model_path'],
    'messages': [{'role': 'user', 'content': 'What is 2 + 2? Answer briefly.'}],
    'temperature': 0, 'max_tokens': 32,
    'chat_template_kwargs': {'enable_thinking': False},
}
chat = post('/v1/chat/completions', chat_payload)
save('chat', chat)
assert chat['choices'][0]['message']['content'], chat

chunks = []
finished = False
with session.post(base + '/v1/chat/completions', json={**chat_payload, 'stream': True}, stream=True, timeout=120) as response:
    response.raise_for_status()
    response.encoding = 'utf-8'
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: '):
            continue
        data = line[6:]
        if data == '[DONE]':
            finished = True
            break
        chunks.append(json.loads(data))
save('chat-stream', chunks)
assert finished, 'SSE 缺少 [DONE]'
stream_text = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks if c.get('choices'))
assert stream_text, chunks
summary = {
    'url': base, 'health': health.status_code,
    'cold_cached_tokens': cold['meta_info']['cached_tokens'],
    'warm_cached_tokens': warm['meta_info']['cached_tokens'],
    'completion_tokens': warm['meta_info']['completion_tokens'],
    'deterministic_output_equal': cold['output_ids'] == warm['output_ids'],
    'generate_text': warm['text'],
    'chat_text': chat['choices'][0]['message']['content'],
    'stream_text': stream_text, 'stream_done': finished,
}
save('summary', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f'原始响应已保存：{out}')
