# Full attention prefill KV cache skip（2026-11-16）

## 1. 假设

`AscendAttentionBackendImpl.forward` 在 `key/value` 非空时总是调用 `reshape_and_cache`。但决策服务是 **pooling + PrefillNoCache**：

- `_get_fia_params` 在 `PrefillNoCache` 下使用当前 K/V、`block_table=None`；
- 没有 decode、没有 prefix cache、没有 chunked prefill；
- 因此这次 `reshape_and_cache` 写入的 KV **无人读取**，理论上可以跳过。

## 2. 实现

在 vLLM-Ascend `attention_v1.py` 中加环境开关：

```python
def _clef_skip_prefill_cache(self, attn_metadata) -> bool:
    import os
    if os.environ.get("CLEF_ATTN_SKIP_PREFILL_CACHE", "0") != "1":
        return False
    return (
        getattr(attn_metadata, "model_runner_type", None) == "pooling"
        and getattr(attn_metadata, "attn_state", None)
        == AscendAttentionState.PrefillNoCache
        and getattr(attn_metadata, "num_decode_tokens", 0) == 0
        and getattr(self, "kv_sharing_target_layer_name", None) is None
        and not getattr(self, "is_kv_producer", False)
    )
```

在 `key/value` 分支里：

```python
if self._clef_skip_prefill_cache(attn_metadata):
    pass
elif attn_metadata.uses_pcp_sharded_layout:
    ...
else:
    query, key, value, output_padded = self.reshape_and_cache(...)
```

默认 `CLEF_ATTN_SKIP_PREFILL_CACHE=0`；实验服务设为 1。guard 排除 KV sharing、KV producer、decode 和 non-pooling。

## 3. 结果

服务：TP2/BF16/eager/4096/seqs1/BIC；30 次正式 + 10 预热，C1。同窗口 no-skip 对照：

| 请求 | 配置 | mean | p50 | p95 | dec/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 307 | skip | 129.09 | 130.18 | 141.12 | 7.742 |
| 307 | **control** | **125.06** | **123.36** | **140.77** | **7.991** |
| 969 | skip | **170.85** | 170.74 | **174.44** | **5.849** |
| 969 | control | 172.51 | 172.00 | 175.50 | 5.790 |

- 307：skip **mean +3.2%**，p95 +0.25%，更慢；
- 969：skip mean −1.0%、p95 −0.6%，在噪声内；
- 冻结 8 例 **8/8**，输出逐位一致（skip 没有改变数值）。

## 4. 结论

- cache 写入在语义上确实是死代码，但它**不是可测量的瓶颈**：可能已被流/其他 kernel 掩盖，或者写入本身很便宜；
- 没有达到 S1 ≥3% 的决策门；307 甚至变慢，969 收益在噪声内；
- **不采用**，恢复 `attention_v1.py` 原始 SHA-256 `a221dd76…`；
- 该方向不再投入；full attention 的剩余空间主要是跨层布局/融合，而不是单点 cache 写回。

## 5. 状态与证据

- 报告证据：`Clef-27B/logs/gdn-data-path-20261115/attn-skip-test/`（service log、frozen8、skip/control 307/969）
- 恢复：`.../restore-attn-skip/`，baseline **PID 706631**，argv/cwd/env 与 phase-0 一致，`/health` 200，冻结 8 例 **8/8**
