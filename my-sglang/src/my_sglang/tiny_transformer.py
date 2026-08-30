"""可运行的 NumPy 单层 decoder，用来闭环 scheduler 与 Transformer 数据流。

这个模块追求接口和因果关系清楚，不追求模型质量或 CPU 性能。它让
``ForwardBatch`` 的展平 token、request row 和物理 KV slot 真正参与一次 forward。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from my_sglang.models import ForwardBatch
from my_sglang.pools import ReqToTokenPool


@dataclass(frozen=True)
class TinyTransformerConfig:
    """单层教学模型的最小配置。"""

    vocab_size: int = 256
    hidden_size: int = 32
    num_heads: int = 4
    intermediate_size: int = 64
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_base: float = 10_000.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.intermediate_size <= 0 or self.max_position_embeddings <= 0:
            raise ValueError(
                "intermediate_size and max_position_embeddings must be positive"
            )
        if self.rms_norm_eps <= 0 or self.rope_base <= 0:
            raise ValueError("rms_norm_eps and rope_base must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def rms_norm(
    values: np.ndarray, weight: np.ndarray, eps: float
) -> np.ndarray:
    """沿最后一维做 RMSNorm，并使用 FP32 累加。"""

    values_fp32 = np.asarray(values, dtype=np.float32)
    weight_fp32 = np.asarray(weight, dtype=np.float32)
    mean_square = np.mean(values_fp32 * values_fp32, axis=-1, keepdims=True)
    return values_fp32 * np.float32(1.0) / np.sqrt(mean_square + eps) * weight_fp32


def silu(values: np.ndarray) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x)。"""

    values_fp32 = np.asarray(values, dtype=np.float32)
    return values_fp32 / (np.float32(1.0) + np.exp(-values_fp32))


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """先减最大值，避免指数溢出的 Softmax。"""

    values_fp32 = np.asarray(values, dtype=np.float32)
    shifted = values_fp32 - np.max(values_fp32, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=axis, keepdims=True)


def apply_rope(
    values: np.ndarray, position: int, base: float = 10_000.0
) -> np.ndarray:
    """对 ``[num_heads, head_dim]`` 的相邻偶/奇特征对应用 RoPE。"""

    values_fp32 = np.asarray(values, dtype=np.float32)
    if values_fp32.ndim != 2 or values_fp32.shape[1] % 2:
        raise ValueError("RoPE input must have shape [num_heads, even head_dim]")
    head_dim = values_fp32.shape[1]
    frequency = np.float32(1.0) / np.power(
        np.float32(base),
        np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim),
    )
    angle = np.float32(position) * frequency
    cosine = np.cos(angle)
    sine = np.sin(angle)
    even = values_fp32[:, 0::2]
    odd = values_fp32[:, 1::2]
    rotated = np.empty_like(values_fp32)
    rotated[:, 0::2] = even * cosine - odd * sine
    rotated[:, 1::2] = even * sine + odd * cosine
    return rotated


class TinyTransformerModel:
    """单层 pre-norm decoder block，以及按物理 slot 保存的 K/V cache。"""

    def __init__(
        self,
        config: TinyTransformerConfig | None = None,
        *,
        trace: list[str] | None = None,
    ) -> None:
        self.config = config or TinyTransformerConfig()
        self.trace = trace if trace is not None else []
        rng = np.random.default_rng(self.config.seed)
        scale = np.float32(1.0 / np.sqrt(self.config.hidden_size))

        def weight(shape: tuple[int, ...]) -> np.ndarray:
            return rng.normal(0.0, scale, size=shape).astype(np.float32)

        hidden = self.config.hidden_size
        intermediate = self.config.intermediate_size
        self.embedding = weight((self.config.vocab_size, hidden))
        self.attn_norm_weight = np.ones((hidden,), dtype=np.float32)
        self.ffn_norm_weight = np.ones((hidden,), dtype=np.float32)
        self.final_norm_weight = np.ones((hidden,), dtype=np.float32)
        self.wq = weight((hidden, hidden))
        self.wk = weight((hidden, hidden))
        self.wv = weight((hidden, hidden))
        self.wo = weight((hidden, hidden))
        self.w_gate = weight((hidden, intermediate))
        self.w_up = weight((hidden, intermediate))
        self.w_down = weight((intermediate, hidden))

        cache_shape = (0, self.config.num_heads, self.config.head_dim)
        self.key_cache = np.empty(cache_shape, dtype=np.float32)
        self.value_cache = np.empty(cache_shape, dtype=np.float32)
        self.final_hidden_cache = np.empty((0, hidden), dtype=np.float32)
        self.kv_valid = np.zeros((0,), dtype=bool)
        self.hidden_valid = np.zeros((0,), dtype=bool)

    def forward_batch(
        self, forward: ForwardBatch, req_to_token_pool: ReqToTokenPool
    ) -> list[int]:
        """执行一批 EXTEND/DECODE，并为每个请求 greedy 采样一个 token。"""

        self._validate_forward(forward)
        sampled: list[int] = []
        for request_index, req in enumerate(forward.reqs):
            token_ids = forward.input_ids_by_req[request_index]
            output_slots = forward.out_cache_loc_by_req[request_index]
            row = forward.req_pool_indices[request_index]
            seq_len = forward.seq_lens[request_index]
            start = forward.extend_range_starts[request_index]
            logits: np.ndarray | None = None

            if row < 0 or row >= req_to_token_pool.size:
                raise ValueError(f"request row {row} is outside ReqToTokenPool")
            if start < 0 or seq_len != start + len(token_ids):
                raise ValueError(
                    "seq_len must equal extend_range_start plus this request's "
                    "input token count"
                )

            for local_index, (token_id, slot) in enumerate(
                zip(token_ids, output_slots, strict=True)
            ):
                position = start + local_index
                history_slots = req_to_token_pool.row(row, position + 1)
                logits = self._forward_token(
                    int(token_id), position, int(slot), history_slots
                )
                self.trace.append(
                    f"model:token:{req.rid}:pos={position}:slot={int(slot)}"
                )

            if logits is None:
                logits = self._logits_from_cached_last_hidden(
                    req.rid, row, seq_len, req_to_token_pool
                )

            next_token = int(np.argmax(logits))
            sampled.append(next_token)
            self.trace.append(
                f"model:sample:{forward.forward_mode.value}:{req.rid}:{next_token}"
            )
        return sampled

    def _forward_token(
        self,
        token_id: int,
        position: int,
        slot: int,
        history_slots: np.ndarray,
    ) -> np.ndarray:
        self._validate_token_and_position(token_id, position)
        if slot < 0:
            raise RuntimeError("output KV slot must be non-negative")
        if len(history_slots) != position + 1 or int(history_slots[-1]) != slot:
            raise RuntimeError("request row does not map the current position to output slot")
        if bool(np.any(history_slots < 0)):
            raise RuntimeError("attention history contains an unmapped KV slot")

        hidden = self.embedding[token_id].copy()
        normed = rms_norm(
            hidden, self.attn_norm_weight, self.config.rms_norm_eps
        )
        query = (normed @ self.wq).reshape(
            self.config.num_heads, self.config.head_dim
        )
        key = (normed @ self.wk).reshape(
            self.config.num_heads, self.config.head_dim
        )
        value = (normed @ self.wv).reshape(
            self.config.num_heads, self.config.head_dim
        )
        query = apply_rope(query, position, self.config.rope_base)
        key = apply_rope(key, position, self.config.rope_base)

        self._ensure_cache_capacity(slot + 1)
        self.key_cache[slot] = key
        self.value_cache[slot] = value
        self.kv_valid[slot] = True
        self.trace.append(f"model:kv_write:pos={position}:slot={slot}")

        history = np.asarray(history_slots, dtype=np.int64)
        if not bool(self.kv_valid[history].all()):
            raise RuntimeError("attention tried to read a KV slot before it was written")
        self.trace.append(
            "model:kv_read:pos="
            f"{position}:slots={','.join(str(int(item)) for item in history)}"
        )
        keys = self.key_cache[history]
        values = self.value_cache[history]
        scores = np.einsum("hd,lhd->hl", query, keys)
        scores /= np.float32(np.sqrt(self.config.head_dim))
        probabilities = stable_softmax(scores, axis=-1)
        context = np.einsum("hl,lhd->hd", probabilities, values)
        hidden = hidden + context.reshape(self.config.hidden_size) @ self.wo

        ffn_input = rms_norm(
            hidden, self.ffn_norm_weight, self.config.rms_norm_eps
        )
        gated = silu(ffn_input @ self.w_gate) * (ffn_input @ self.w_up)
        hidden = hidden + gated @ self.w_down
        self.final_hidden_cache[slot] = hidden
        self.hidden_valid[slot] = True
        return self._logits(hidden)

    def _logits_from_cached_last_hidden(
        self,
        rid: str,
        row: int,
        seq_len: int,
        req_to_token_pool: ReqToTokenPool,
    ) -> np.ndarray:
        if seq_len <= 0:
            raise RuntimeError("cannot sample from an empty sequence")
        last_slot = req_to_token_pool.get(row, seq_len - 1)
        if last_slot is None or last_slot >= len(self.hidden_valid):
            raise RuntimeError("full prefix hit has no cached final hidden state")
        if not bool(self.hidden_valid[last_slot]):
            raise RuntimeError("full prefix hit refers to an unwritten hidden-state slot")
        self.trace.append(f"model:full_prefix_hit:{rid}:slot={last_slot}")
        return self._logits(self.final_hidden_cache[last_slot])

    def _logits(self, hidden: np.ndarray) -> np.ndarray:
        normalized = rms_norm(
            hidden, self.final_norm_weight, self.config.rms_norm_eps
        )
        # 教学模型使用 tied embeddings：LM head weight 就是 embedding table。
        return normalized @ self.embedding.T

    def _ensure_cache_capacity(self, size: int) -> None:
        current = len(self.kv_valid)
        if size <= current:
            return
        grow = size - current
        kv_tail = np.zeros(
            (grow, self.config.num_heads, self.config.head_dim),
            dtype=np.float32,
        )
        hidden_tail = np.zeros(
            (grow, self.config.hidden_size), dtype=np.float32
        )
        self.key_cache = np.concatenate((self.key_cache, kv_tail), axis=0)
        self.value_cache = np.concatenate((self.value_cache, kv_tail.copy()), axis=0)
        self.final_hidden_cache = np.concatenate(
            (self.final_hidden_cache, hidden_tail), axis=0
        )
        self.kv_valid = np.concatenate((self.kv_valid, np.zeros(grow, dtype=bool)))
        self.hidden_valid = np.concatenate(
            (self.hidden_valid, np.zeros(grow, dtype=bool))
        )

    def _validate_forward(self, forward: ForwardBatch) -> None:
        request_fields = (
            forward.req_pool_indices,
            forward.seq_lens,
            forward.extend_seq_lens,
            forward.extend_range_starts,
        )
        if any(len(field) != forward.batch_size for field in request_fields):
            raise ValueError("ForwardBatch request fields must match batch_size")
        if len(forward.input_ids) != len(forward.out_cache_loc):
            raise ValueError("input_ids and out_cache_loc must have the same length")
        if sum(forward.extend_seq_lens) != len(forward.input_ids):
            raise ValueError("extend_seq_lens must partition flat input_ids")

    def _validate_token_and_position(self, token_id: int, position: int) -> None:
        if token_id < 0 or token_id >= self.config.vocab_size:
            raise ValueError(
                f"token id {token_id} is outside vocab_size={self.config.vocab_size}"
            )
        if position < 0 or position >= self.config.max_position_embeddings:
            raise ValueError(
                f"position {position} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )


class TinyTransformerRunner:
    """把教学模型适配成同步 ``MiniScheduler`` runner。"""

    def __init__(self, config: TinyTransformerConfig | None = None) -> None:
        self.trace: list[str] = []
        self.model = TinyTransformerModel(config, trace=self.trace)

    def run_batch(
        self, forward: ForwardBatch, req_to_token_pool: ReqToTokenPool
    ) -> list[int]:
        return self.model.forward_batch(forward, req_to_token_pool)

    def remove_request(self, req_id: str) -> None:
        # K/V 是否保留由 allocator/radix cache 的 slot 所有权决定；runner 不能按
        # req_id 清空共享 prefix。
        self.trace.append(f"model:remove_request:{req_id}")
