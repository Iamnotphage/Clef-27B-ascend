#!/usr/bin/env python3
"""Cross-check the pinned EvalScope ERQA adapter on the fixed smoke subset.

Registers a local ERQA benchmark over the official snapshot, runs the native
System One decision path with images, and compares every predicted letter with
the project's direct /v1/decide smoke run.  The service exposes the same
StructuredDecisionRequest handler on /v1/systemone and /v1/decide; the formal
run uses /v1/decide, while EvalScope's SystemOne provider posts to
/v1/systemone.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

from erqa_baseline import smoke_selection
from erqa_protocol import load_samples, parse_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path, required=True,
                        help="项目直接 /v1/decide smoke 运行目录")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples, _ = load_samples(args.data_dir)
    by_id = {sample["id"]: sample for sample in samples}
    selected = smoke_selection(samples)
    if args.limit:
        selected = selected[: args.limit]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.work_dir / "erqa-smoke.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for sample in selected:
            record = {
                "question_id": sample["id"],
                "question": sample["question"],
                "question_type": sample["question_type"],
                "answer": sample["answer"],
                "visual_indices": sample["visual_indices"],
                "images": [str((args.data_dir / img["path"]).resolve()) for img in sample["images"]],
                "options": [list(option) for option in parse_options(sample["question"])],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    from evalscope import TaskConfig, run_task
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.api.metric import MetricSelector
    from evalscope.api.registry import register_benchmark
    from evalscope.benchmarks.erqa.erqa_adapter import ERQAAdapter, SUBSET_LIST

    def local_loader(adapter, **kwargs):
        from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
        return DefaultDataAdapter.load_from_disk(adapter, use_local_loader=True)

    @register_benchmark(BenchmarkMeta(
        name="clef_erqa_smoke19",
        pretty_name="Clef ERQA official smoke subset",
        description="Official ERQA TFRecord 5bba488, fixed 19-sample smoke subset, native System One multimodal decision path.",
        dataset_id=str(records_path.resolve()),
        eval_split="test", train_split=None, subset_list=SUBSET_LIST,
        metric_list=["accuracy"],
        primary_metric=MetricSelector(name="accuracy", aggregation="mean"),
        supports_choice=True,
    ))
    class LocalERQAAdapter(ERQAAdapter):
        load_from_disk = local_loader

    run_task(TaskConfig(
        model="clef", eval_type="systemone_api", api_url=args.api_url, api_key="EMPTY",
        datasets=["clef_erqa_smoke19"],
        eval_batch_size=1, generation_config={"timeout": 600, "retries": 2},
        work_dir=str(args.work_dir.resolve()), no_timestamp=True,
    ))

    prediction_files = glob.glob(str(args.work_dir / "predictions" / "**" / "*.jsonl"), recursive=True)
    if not prediction_files:
        raise SystemExit("EvalScope did not write a predictions JSONL")
    evalscope = {}
    for path in prediction_files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            sample_id = metadata.get("question_id")
            if sample_id not in {sample["id"] for sample in selected}:
                continue
            content = (
                ((row.get("model_output") or {}).get("choices") or [{}])[0]
                .get("message", {}).get("content", "")
            )
            prediction = content.replace("ANSWER:", "").strip() or None
            response = (
                ((row.get("model_output") or {}).get("metadata") or {})
                .get("choice_response") or {}
            )
            answer = (response.get("answers") or {}).get("answer") or {}
            evalscope[sample_id] = {
                "prediction": answer.get("choice") or prediction,
                "probabilities": answer.get("probabilities"),
                "file": path,
            }
    direct = {
        row["id"]: row for row in (
            json.loads(line) for line in
            (args.direct_run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip())
        if row["id"] in {sample["id"] for sample in selected}
    }
    mismatches = []
    for sample in selected:
        left = evalscope.get(sample["id"], {}).get("prediction")
        right = direct.get(sample["id"], {}).get("prediction")
        if left != right:
            mismatches.append({"id": sample["id"], "evalscope": left, "direct": right})
    report = {
        "samples": len(selected),
        "evalscope_predictions": len(evalscope),
        "direct_predictions": len(direct),
        "mismatches": mismatches,
        "equal": not mismatches and len(evalscope) == len(selected),
        "files": prediction_files,
    }
    (args.work_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
