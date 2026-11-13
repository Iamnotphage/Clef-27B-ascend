"""Launch the pinned vLLM-Ascend structured-decisions service."""

import argparse
import json
import os
from pathlib import Path
import sys


def model_files(model_dir):
    """Check both the joint head and every shard named by the backbone index."""
    names = {
        "config.json",
        "tokenizer_config.json",
        "joint_head_config.json",
        "joint_head.safetensors",
        "model.safetensors.index.json",
    }
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("主干权重索引为空")
    for shard in shards:
        path = (model_dir / shard).resolve()
        if not path.is_relative_to(model_dir):
            raise ValueError(f"权重索引包含模型目录外的路径：{shard}")
    names.update(shards)
    missing = sorted(
        name for name in names
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    )
    if missing:
        raise ValueError(f"模型文件缺失或为空：{', '.join(missing)}")
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(
        description="启动 Clef-27B 的 vLLM-Ascend /v1/systemone 服务。",
        epilog="先加载 scripts/activate_env.sh；附加 vLLM 参数放在 -- 之后。",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/workspace/yangchen/models/Clef-27B"),
        help="完整模型目录，可通过 MODEL_DIR 设置默认值。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=None,
        help="每批调度的 token 上限；省略时与 --max-model-len 保持一致。",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dry-run", action="store_true", help="输出启动参数 JSON。")
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 应在 1 到 65535 之间")
    if min(args.tensor_parallel_size, args.max_model_len, args.max_num_seqs) < 1:
        parser.error("并行数、上下文长度和序列数应为正数")
    extra = args.vllm_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    passthrough_batched_tokens = "--max-num-batched-tokens" in extra
    if passthrough_batched_tokens and args.max_num_batched_tokens is not None:
        parser.error("--max-num-batched-tokens 重复指定：只允许启动参数或 -- 之后透传其一")
    max_num_batched_tokens = args.max_num_batched_tokens
    if max_num_batched_tokens is None and not passthrough_batched_tokens:
        max_num_batched_tokens = args.max_model_len
    if max_num_batched_tokens is not None:
        if max_num_batched_tokens < 1:
            parser.error("--max-num-batched-tokens 应为正数")
        if max_num_batched_tokens < args.max_model_len:
            parser.error(
                "--max-num-batched-tokens 不应小于 --max-model-len（chunked prefill 已关闭，"
                f"单请求最多 {args.max_model_len} tokens）"
            )
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization 应在 (0, 1] 之间")
    model_dir = Path(args.model_dir).expanduser().resolve()
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir),
        "--served-model-name", "clef",
        "--runner", "pooling",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--enforce-eager",
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
    ]
    if not passthrough_batched_tokens:
        command += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    command += [
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--skip-mm-profiling",
        "--host", args.host,
        "--port", str(args.port),
        *extra,
    ]
    launch = {
        "python": sys.executable, "model_dir": str(model_dir), "argv": command,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_batched_tokens_source": (
            "passthrough" if passthrough_batched_tokens
            else "explicit" if args.max_num_batched_tokens is not None
            else "default_follows_max_model_len"
        ),
    }
    if args.dry_run:
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        return
    try:
        launch["model_files"] = model_files(model_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"完整模型检查失败：{exc}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
    print(json.dumps(launch, ensure_ascii=False), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
