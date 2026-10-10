# Token、字节、UTF-8 与 BPE：从文字到 embedding

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

token 是 tokenizer 词表中的一个单位；模型通常拿到的是它的整数 ID，再按 ID 查 embedding 向量。token 不是一个固定字节数，也不一定是完整字符或单词。

## 四层表示

```text
可见文字 “你”
    ↓ Unicode 码点 U+4F60
UTF-8 字节 E4 BD A0
    ↓ tokenizer 的规则和词表
一个或多个 token ID
    ↓ embedding 查表
一个 ID 对应一行向量（维度由模型定义）
```

| 单位 | 是什么 | “你”的例子 |
|---|---|---|
| 可见字符/字素 | 用户眼里的一份显示单位 | 普通汉字是一份；某些 emoji 由多个码点组合 |
| Unicode 码点 | 字符编码空间中的编号 | U+4F60 |
| byte / 字节 | 8 bit 存储单位 | UTF-8 用 3 个字节 |
| token | 词表单位 | 取决于 tokenizer，可 1 个也可多个 |
| token ID | 词表索引 | 不是 U+4F60 或 UTF-8 字节值 |
| embedding | 模型学习的向量 | Qwen3-0.6B hidden_size=1024 |

## BPE 是什么意思

Byte Pair Encoding：反复把常见的相邻符号对合并成新的单位。byte-level BPE 的初始符号来自字节表示，但预切分、特殊 token 和训练细节也影响结果。

```text
玩具语料中经常出现：l o w
初始单位：          l | o | w
学习合并：          l + o → lo
进一步合并：        lo + w → low
最终分词：          low | er    （仅示意，不是 Qwen 实测）
```

BPE 训练学习的是压缩/分段的合并规则；“概念”主要由后续语言模型通过上下文训练学习。不能说每个字节在 BPE 阶段就学会了完整语义。

## 为什么切开 UTF-8 仍有意义

| 好处 | 代价 |
|---|---|
| 字节单位能覆盖任意可编码文本，降低未知字符问题 | 冷门字符可能分成更多 token |
| 常见片段能合并成更长单位 | 单个 token 有时不是完整 UTF-8 字符 |
| 同一片段在不同词/上下文中复用 | 解码需要考虑边界与上下文 |

不是“每三个 UTF-8 字节先组成一个字符向量”，也不是“每个字节必定单独一个向量”。词表里可有单字节片段，也可有多个字节合成的片段；模型按最终 token ID 查向量。

## 增量解码为什么不能逐 token 拼字符串

以玩具词表为例：token a 对应 `E4 BD`，token b 对应 `A0`。

```text
decode([a])：字节未完整，可能是空串/替换字符（依实现）
decode([b])：孤立 continuation byte，也不完整
decode([a,b])：E4 BD A0 → “你”
```

所以不能假定 `decode([a]) + decode([b]) == decode([a,b])`。空格清理、特殊 token 和上下文规则也可能影响结果。

### 怎样知道 UTF-8 完整

| 首字节模式 | 总字节数 |
|---|---|
| 0xxxxxxx | 1 |
| 110xxxxx | 2 |
| 1110xxxx | 3 |
| 11110xxx | 4 |

后续字节必须是 `10xxxxxx`，还要检查合法范围，排除 overlong、代理码点和超出 U+10FFFF。Python 增量 decoder 会处理这些规则：

```python
import codecs
d = codecs.getincrementaldecoder('utf-8')()
assert d.decode(bytes.fromhex('e4 bd'), final=False) == ''
assert d.decode(bytes.fromhex('a0'), final=False) == '你'
```

这段只是 UTF-8 示范，不是 SGLang 的完整 detokenizer 实现。SGLang 维护已读偏移和上下文，按 tokenizer 解码结果推进，只输出可确认的新部分。UTF-8 字节完整也不等于上层文本处理永远无需上下文。

## 当前 SGLang 怎样推进输出

`DetokenizerManager._decode_batch_token_id_output` 的主干如下：

```text
同一 rid 累积 decode_ids
  → 解码 [surr_offset : read_offset] 的上下文文本
  → 解码 [surr_offset : 最新末尾] 的完整候选文本
  → 用上下文文本长度切出 new_text
       ├─ 非空、结尾不是 �：提交文本，推进 token 偏移
       └─ 不完整候选：只发可打印前缀，保持 token 偏移，下轮重试
  → sent_offset 去掉已经发过的字符，避免重复
```

结束时还会处理匹配到的 stop 并输出剩余尾部。这里不是直接在 SRT 中逐字节执行上面的 UTF-8 decoder，而是依靠 tokenizer 解码结果、替换字符检查、可打印文本和偏移维护增量输出。合法字符串本身也可能包含 `�`；这是一套具体实现策略，不是 UTF-8 合法性的全部定义。

| 字段 | 作用 |
|---|---|
| surr_offset | 重解码时保留的 token 上下文起点 |
| read_offset | 已确认读过的 token 边界 |
| decoded_text_len | 已提交文本的字符长度 |
| sent_offset | 已发给客户端的位置，可暂时领先于提交长度 |

## 用自己的模型看实际 token

```bash
python/.venv/bin/python - <<'PY'
from transformers import AutoTokenizer
from pathlib import Path
t = AutoTokenizer.from_pretrained(Path.home()/'.modelscope/models/Qwen3-0.6B')
for s in ['你', '你好', 'lower', '🙂']:
    ids = t.encode(s, add_special_tokens=False)
    print(repr(s), s.encode('utf-8').hex(), ids, t.convert_ids_to_tokens(ids), repr(t.decode(ids)))
PY
```

`convert_ids_to_tokens` 可能显示 tokenizer 的内部字节替代符号，不要把那个显示字符串直接当成最终文字。

chat template 还会添加角色、分隔和模型特定控制 token，所以聊天请求的 token 数不等于只编码用户正文得到的数。

## 本次本地 tokenizer 实测

| 文字 | UTF-8 hex | Qwen3-0.6B token IDs | 逐个 decode 后拼接 | 整段 decode |
|---|---|---|---|---|
| 你 | e4bda0 | [56568] | 你 | 你 |
| 你好 | e4bda0e5a5bd | [108386] | 你好 | 你好 |
| lower | 6c6f776572 | [14772] | lower | lower |
| 🙂 | f09f9982 | [145080] | 🙂 | 🙂 |
| 龘 | e9be98 | [82912, 246] | �� | 龘 |
| 𪚥 | f0aa9aa5 | [100130, 248, 98] | ��� | 𪚥 |

这组结果直接说明：常见词片段、两个汉字或 emoji 可以合成一个 token；某些字符又会拆成多个 token，单独 decode 得到替换字符，合起来才恢复原字。ID 与具体模型词表相关，不可推广成所有 tokenizer 的固定编号。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | 编码入口 |
| [python/sglang/srt/managers/detokenizer_manager.py](../../python/sglang/srt/managers/detokenizer_manager.py) | 偏移、上下文与增量输出 |
| [python/sglang/srt/entrypoints/openai/chat_encoding.py](../../python/sglang/srt/entrypoints/openai/chat_encoding.py) | chat 编码 |
| [rust/sglang-processor/src](../../rust/sglang-processor/src) | 另一条处理实现 |
