#!/usr/bin/env python3
"""Fixed development subsets, pinned EvalScope scorers, direct local JSONL loading.

The two local adapter names keep these subset reports identifiable. Their score
and aggregation methods are inherited from the pinned native benchmarks; only
the disk loader follows GeneralDecisionAdapter's built-in JSONL loading path.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--benchmark", action="append", choices=("jevbench", "open_decisions"))
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    full = args.scope == "full"
    jev_name = "clef_jevbench_public231" if full else "clef_jevbench_smoke"
    open_name = "clef_open_decisions_full2352" if full else "clef_open_decisions_smoke"
    jev_file = "jevbench-public231.jsonl" if full else "jevbench-smoke12.jsonl"
    open_file = "open-decisions2352.jsonl" if full else "open_decisions-smoke12.jsonl"

    from evalscope import TaskConfig, run_task
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.api.metric import MetricSelector
    from evalscope.api.registry import register_benchmark
    from evalscope.benchmarks.jevbench.jevbench_adapter import JevbenchAdapter
    from evalscope.benchmarks.open_decisions.open_decisions_adapter import OpenDecisionsAdapter

    def local_loader(adapter, **kwargs):
        # Equivalent to GeneralDecisionAdapter.load_from_disk at the fixed SHA.
        from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
        return DefaultDataAdapter.load_from_disk(adapter, use_local_loader=True)

    @register_benchmark(BenchmarkMeta(
        name=jev_name,
        pretty_name="Clef JevBench fixed public text, 231 records" if full else "Clef JevBench fixed 12-record smoke subset",
        description=(
            "JevBench fixed public text snapshot 609a664e81f46e03731600bd7bd9f1b7a61257c5; "
            + ("all 231 public records" if full else "12 fixed development smoke records")
            + "; original rubrics and candidate order, native JevBench scoring, direct local JSONL loading."
        ),
        dataset_id=str((args.data_dir / jev_file).resolve()),
        eval_split="test", train_split=None, subset_list=["default"],
        metric_list=["public_accuracy"],
        primary_metric=MetricSelector(name="public_accuracy", aggregation="official"),
    ))
    class LocalJevbenchAdapter(JevbenchAdapter):
        load_from_disk = local_loader

    @register_benchmark(BenchmarkMeta(
        name=open_name,
        pretty_name="Clef OpenDecisions complete fixed 2352-record input" if full else "Clef OpenDecisions fixed 12-record smoke subset",
        description=(
            "OpenDecisions fixed snapshot ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe; "
            + ("all 2352 source records" if full else "12 fixed development smoke records")
            + "; original rubrics and candidate order, native OpenDecisions scoring, direct local JSONL loading."
        ),
        dataset_id=str((args.data_dir / open_file).resolve()),
        eval_split="test", train_split=None, subset_list=["default"],
        metric_list=["capability"],
        primary_metric=MetricSelector(name="capability", aggregation="official"),
    ))
    class LocalOpenDecisionsAdapter(OpenDecisionsAdapter):
        load_from_disk = local_loader

    chosen = args.benchmark or ["jevbench", "open_decisions"]
    return run_task(TaskConfig(
        model="clef", eval_type="systemone_api", api_url=args.api_url, api_key="EMPTY",
        datasets=[{"jevbench": jev_name, "open_decisions": open_name}[name] for name in chosen],
        eval_batch_size=1, generation_config={"timeout": 300, "retries": 1},
        work_dir=str(args.work_dir.resolve()), no_timestamp=True,
    ))


if __name__ == "__main__":
    main()
