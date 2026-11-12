#!/usr/bin/env python3
"""Summarize exported torch_npu traces without equating kernel sums to latency.

Example:
  python scripts/analyze_systemone_profile.py /absolute/run/traces \
      --expected-ranks 2 --output-dir /absolute/run/analysis --strict

Requires only Python's standard library. Run torch_npu.profiler.profiler.analyse
in a normal independent process before this script when worker export failed.
All timestamps/durations below use microseconds in the exported clock domain.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys


COMM = re.compile(r"hccl|hcom|all[_ ]?reduce|all[_ ]?gather|reduce[_ ]?scatter|all[_ ]?to[_ ]?all|broadcast|(?:^|[/:_ ])(?:send|recv)(?:$|[/:_ ])", re.I)
MARKER = re.compile(r"clef|systemone|joint|scheduler|tokeniz|process_input|encode_request|model_forward|execute_model|pooler", re.I)


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def finite(value):
    try:
        number = float(str(value).replace(",", ""))
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def pick(row, *names):
    indexed = {norm(key): value for key, value in row.items()}
    for name in names:
        value = indexed.get(norm(name))
        if value not in (None, "", "N/A", "NA", "None"):
            return value
    return None


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def merge_intervals(intervals):
    result = []
    for start, end in sorted((a, b) for a, b in intervals if b > a):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def union_us(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_us(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def interval(row):
    start = finite(pick(row, "Start Time(us)", "Task Start Time(us)", "Start Timestamp(us)", "ts"))
    duration = finite(pick(row, "Duration(us)", "Task Duration(us)", "dur"))
    return (start, start + duration) if start is not None and duration is not None and duration >= 0 else None


def kernel_rows(path, root):
    rows, warnings = [], []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        for line, row in enumerate(reader, 2):
            times = interval(row)
            if times is None:
                continue
            name = pick(row, "Name", "Op Name", "OP Name", "Kernel Name") or "<unnamed>"
            op_type = pick(row, "Type", "OP Type", "Op Type", "Task Type") or ""
            rows.append({
                "name": name, "op_type": op_type,
                "start_us": times[0], "duration_us": times[1] - times[0],
                "end_us": times[1], "communication": bool(COMM.search(name + " " + op_type)),
                "shape": pick(row, "Input Shapes", "Input Shape", "Shapes"),
                "dtype": pick(row, "Input Data Types", "Input Data Type", "Input Dtypes", "Dtype"),
                "output_shape": pick(row, "Output Shapes", "Output Shape"),
                "stream": pick(row, "Stream Id", "Stream ID", "Stream"),
                "device": pick(row, "Device_id", "Device ID", "Device Id"),
                "task_id": pick(row, "Task Id", "Task ID"),
                "evidence_file": str(path.relative_to(root)), "evidence_line": line,
            })
    if not rows:
        warnings.append(f"No timed kernels parsed from {path.name}; headers={fields}")
    return rows, fields, warnings


def event_duration(event):
    ts, duration = finite(event.get("ts")), finite(event.get("dur"))
    return (ts, ts + duration) if ts is not None and duration is not None and duration >= 0 else None


def trace_rows(path, root):
    data = read_json(path)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    process_names, thread_names = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        args = event.get("args", {})
        if event.get("name") == "process_name":
            process_names[str(event.get("pid"))] = str(args.get("name", ""))
        if event.get("name") == "thread_name":
            thread_names[(str(event.get("pid")), str(event.get("tid")))] = str(args.get("name", ""))
    timed, kernels, scopes = [], [], []
    categories = Counter()
    for index, event in enumerate(events):
        categories[str(event.get("cat", ""))] += 1
        times = event_duration(event)
        if event.get("ph") != "X" or times is None:
            continue
        name, category = str(event.get("name", "")), str(event.get("cat", ""))
        process = process_names.get(str(event.get("pid")), "")
        thread = thread_names.get((str(event.get("pid")), str(event.get("tid"))), "")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        row = {
            "name": name, "op_type": category,
            "start_us": times[0], "duration_us": times[1] - times[0], "end_us": times[1],
            "communication": bool(COMM.search(name + " " + category)),
            "shape": pick(args, "Input Dims", "Input Shapes", "Input Shape"),
            "dtype": pick(args, "Input type", "Input Data Types", "Input Dtypes"),
            "stream": thread or event.get("tid"), "device": process,
            "evidence_file": str(path.relative_to(root)), "evidence_event_index": index,
        }
        timed.append(row)
        # Metadata is essential: CPU-side hccl launches are not device kernels.
        on_device = bool(re.search(r"kernel|ascend.?hardware|npu.?hardware", category + " " + process, re.I))
        if on_device and not re.search(r"runtime|cpu_op|python|torch", category, re.I):
            kernels.append(row)
        elif MARKER.search(name):
            scopes.append(dict(row, pid=event.get("pid"), tid=event.get("tid"), category=category))
    return kernels, scopes, {
        "event_count": len(events), "timed_event_count": len(timed),
        "device_kernel_event_count": len(kernels),
        "categories": dict(categories), "process_names": process_names,
        "timed_event_range_us": [min(r["start_us"] for r in timed), max(r["end_us"] for r in timed)] if timed else None,
    }


def summarize_rows(rows, capture_span=None):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["name"], row.get("op_type", ""), json.dumps(row.get("shape"), sort_keys=True), json.dumps(row.get("dtype"), sort_keys=True))
        grouped[key].append(row)
    result = []
    for (name, op_type, shape, dtype), group in grouped.items():
        times = [(row["start_us"], row["end_us"]) for row in group]
        cumulative = sum(row["duration_us"] for row in group)
        covered = union_us(times)
        result.append({
            "name": name, "op_type": op_type, "shape": json.loads(shape), "dtype": json.loads(dtype),
            "count": len(group), "cumulative_us": cumulative, "mean_us": cumulative / len(group),
            "max_us": max(row["duration_us"] for row in group), "interval_union_us": covered,
            "capture_window_coverage_fraction": covered / capture_span if capture_span else None,
            "critical_path_contribution_us": None,
            "communication": any(row["communication"] for row in group),
            "evidence": [{key: row[key] for key in ("evidence_file", "evidence_line", "evidence_event_index", "start_us", "end_us", "stream") if key in row} for row in group[:3]],
        })
    return sorted(result, key=lambda row: row["cumulative_us"], reverse=True)


def communication_json(path, root):
    result = []

    def microseconds(info, *names):
        value = finite(pick(info, *names))
        return value * 1000 if value is not None else None

    def visit(value, keys):
        if not isinstance(value, dict):
            return
        time_info = next((item for key, item in value.items() if norm(key) == "communicationtimeinfo" and isinstance(item, dict)), None)
        if time_info is not None:
            bandwidth = next((item for key, item in value.items() if norm(key) == "communicationbandwidthinfo"), {})
            result.append({"operation_path": "/".join(keys), "time_info": time_info, "bandwidth_info": bandwidth,
                           "reported_elapsed_us": microseconds(time_info, "Elapse Time(ms)", "Elapsed Time(ms)"),
                           "reported_wait_us": microseconds(time_info, "Wait Time(ms)"),
                           "reported_synchronization_us": microseconds(time_info, "Synchronization Time(ms)"),
                           "evidence_file": str(path.relative_to(root)), "evidence_json_path": keys})
            return
        for key, item in value.items():
            visit(item, keys + [key])

    visit(read_json(path), [])
    return result


def rank_id(path):
    candidates = []
    for info in sorted(path.glob("profiler_info*.json")):
        try:
            data = read_json(info)
            candidates.append(data)
        except (OSError, ValueError):
            pass
    for data in candidates:
        queue = [data]
        while queue:
            value = queue.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if norm(key) in ("rankid", "rank") and isinstance(item, (int, str)) and re.fullmatch(r"\d+", str(item)):
                        return int(item)
                    if isinstance(item, dict):
                        queue.append(item)
    match = re.search(r"(?:rank|worker)[_-]?(\d+)", str(path), re.I)
    return int(match[1]) if match else None


def analyze_rank(path, root, index, max_top):
    warnings = []
    exported = path / "ASCEND_PROFILER_OUTPUT"
    search = exported if exported.is_dir() else path
    kernel_files = sorted(search.glob("*kernel*details*.csv"))
    if not kernel_files:
        kernel_files = sorted(search.glob("*op_summary*.csv"))
    rows, headers = [], {}
    for kernel_file in kernel_files:
        parsed, fields, issues = kernel_rows(kernel_file, root)
        rows.extend(parsed)
        headers[str(kernel_file.relative_to(root))] = fields
        warnings.extend(issues)
    traces = sorted(set(search.glob("*trace*.json")) | set(search.glob("*trace*.json.gz")))
    scopes, timeline_metadata, timeline_kernel_rows = [], [], []
    for trace in traces:
        try:
            kernels, markers, metadata = trace_rows(trace, root)
            timeline_kernel_rows.extend(kernels)
            scopes.extend(markers)
            timeline_metadata.append(dict(metadata, file=str(trace.relative_to(root))))
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"Cannot parse {trace.name}: {type(exc).__name__}: {exc}")
    kernel_source = "kernel CSV" if rows else "timeline device events"
    if not rows:
        rows = timeline_kernel_rows
    all_times = [(row["start_us"], row["end_us"]) for row in rows]
    compute_times = [(row["start_us"], row["end_us"]) for row in rows if not row["communication"]]
    comm_times = [(row["start_us"], row["end_us"]) for row in rows if row["communication"]]
    window = [min(a for a, _ in all_times), max(b for _, b in all_times)] if all_times else None
    span = window[1] - window[0] if window else None
    all_union, compute_union, comm_union = union_us(all_times), union_us(compute_times), union_us(comm_times)
    overlap = intersection_us(compute_times, comm_times)
    operations = summarize_rows(rows, span)
    comm_ops = []
    for comm_file in sorted(search.glob("*communication*.json")):
        try:
            comm_ops.extend(communication_json(comm_file, root))
        except (OSError, ValueError) as exc:
            warnings.append(f"Cannot parse {comm_file.name}: {exc}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    export_files = [item for item in files if "ASCEND_PROFILER_OUTPUT" in item.parts]
    raw_directories = [item for item in path.iterdir() if item.is_dir() and
                       (item.name in ("FRAMEWORK", "framework") or item.name.startswith("PROF_"))]
    has_raw = any(item.is_file() and item.stat().st_size > 0
                  for directory in raw_directories for item in directory.rglob("*"))
    valid_timeline = any(item["device_kernel_event_count"] for item in timeline_metadata)
    completeness = {
        "raw_ascend_pt_directory": path.name.endswith("_ascend_pt"),
        "raw_framework_or_prof_directory": has_raw,
        "export_directory": exported.is_dir(),
        "timed_kernel_count": len(rows), "valid_device_timeline": valid_timeline,
        "valid": path.name.endswith("_ascend_pt") and has_raw and bool(rows) and valid_timeline,
    }
    if any(row.get("shape") for row in rows):
        shape_status = "present in exported kernel rows"
    elif any(row.get("shape") for row in scopes):
        shape_status = "present in CPU scopes; device shapes require correlation"
    else:
        shape_status = "not exported"
    return {
        "rank": rank_id(path), "discovery_index": index, "directory": str(path.resolve()),
        "relative_directory": str(path.relative_to(root)), "completeness": completeness,
        "raw_file_count": len(files) - len(export_files), "total_bytes": sum(item.stat().st_size for item in files),
        "exported_file_index": [{"path": str(item.relative_to(root)), "bytes": item.stat().st_size, "sha256": file_sha(item)} for item in export_files],
        "kernel_source": kernel_source, "kernel_csv_headers": headers, "shape_status": shape_status,
        "kernel_window_us": window, "kernel_window_span_us": span,
        "kernel_cumulative_us": sum(row["duration_us"] for row in rows),
        "kernel_union_us": all_union, "compute_union_us": compute_union, "communication_union_us": comm_union,
        "compute_communication_overlap_us": overlap, "communication_nonoverlapped_us": comm_union - overlap,
        "no_device_kernel_within_window_us": span - all_union if span else None,
        "kernel_count": len(rows), "communication_kernel_count": sum(row["communication"] for row in rows),
        "top_operations": operations[:max_top], "all_operations": operations,
        "communication_operations": [row for row in operations if row["communication"]],
        "communication_details": comm_ops, "cpu_scopes": summarize_rows(scopes, span),
        "cpu_scope_events": scopes,
        "timeline_metadata": timeline_metadata, "warnings": warnings,
        "method": {"busy": "union of exported device kernel intervals in the device clock domain",
                   "communication": "case-insensitive HCCL/collective name classification; inspect operation names",
                   "idle": "gaps between first and last exported kernel; includes inter-request idle and host work",
                   "critical_path": "Not inferred from cumulative kernel duration; request markers and event dependencies required",
                   "shape": "Taken only from explicit exported shapes; no inference from names"},
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def cpu_observations(root):
    """Read opt-in host observations; preserve inclusive/overlapping semantics."""
    events, errors, files = [], [], []
    for path in sorted(root.glob("cpu-*.jsonl")):
        files.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha(path)})
        with path.open(encoding="utf-8") as stream:
            for line, text in enumerate(stream, 1):
                try:
                    value = json.loads(text)
                    if not isinstance(value, dict):
                        raise ValueError("Observation must be a JSON object")
                    start = value["start_wall_ns"]
                    duration = value["duration_ns"]
                    if not isinstance(start, int) or not isinstance(duration, int) or duration < 0:
                        raise ValueError("start_wall_ns and duration_ns must be integer nanoseconds")
                    if not isinstance(value.get("name"), str):
                        raise ValueError("name must be a string")
                    events.append(dict(value, end_wall_ns=start + duration,
                                       evidence_file=str(path.resolve()), evidence_line=line))
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({"path": str(path.resolve()), "line": line, "error": str(exc)})
    groups, requests = defaultdict(list), defaultdict(list)
    for event in events:
        groups[(event["name"], event.get("pid"), event.get("rank"))].append(event)
        if event.get("request_id") is not None:
            requests[str(event["request_id"])].append(event)

    def summary(group):
        intervals = [(event["start_wall_ns"], event["end_wall_ns"]) for event in group]
        cumulative = sum(event["duration_ns"] for event in group)
        union = union_us(intervals)  # Interval arithmetic retains input units.
        return {"count": len(group), "inclusive_cumulative_ms": cumulative / 1e6,
                "mean_ms": cumulative / len(group) / 1e6, "interval_union_ms": union / 1e6,
                "overlap_with_same_name_ms": (cumulative - union) / 1e6,
                "wall_range_ns": [min(a for a, _ in intervals), max(b for _, b in intervals)],
                "evidence": [{"file": event["evidence_file"], "line": event["evidence_line"]} for event in group[:3]]}

    stages = [dict(name=name, pid=pid, rank=rank, **summary(group)) for (name, pid, rank), group in groups.items()]
    stages.sort(key=lambda row: row["inclusive_cumulative_ms"], reverse=True)
    request_index = [{"request_id": request_id, "event_count": len(group),
                      "names": dict(Counter(event["name"] for event in group)),
                      "process_ids": sorted({event.get("pid") for event in group if event.get("pid") is not None}),
                      "wall_range_ns": [min(event["start_wall_ns"] for event in group), max(event["end_wall_ns"] for event in group)],
                      "evidence": [{"file": event["evidence_file"], "line": event["evidence_line"]} for event in group]}
                     for request_id, group in sorted(requests.items())]
    return {"root": str(root.resolve()), "passed": bool(events) and not errors,
            "source_files": files, "event_count": len(events),
            "errors": errors, "stages": stages, "requests": request_index, "events": events,
            "method": {"timing_scope": "host_wall_inclusive; nested stages and concurrent tasks overlap",
                       "end_wall_ns": "start_wall_ns plus monotonic duration_ns; assumes stable wall clock within each scope",
                       "union": "computed per name/pid/rank; stage unions are not additive across names",
                       "device_alignment": "Host wall nanoseconds are retained separately from exported NPU microseconds; clock calibration is required for alignment"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--trace-dir", type=Path, action="append", help="Select a window's raw rank directories; repeat once per rank")
    parser.add_argument("--observations-root", type=Path, help="Read optional profile_instrumentation.py cpu-<pid>.jsonl files")
    parser.add_argument("--max-top", type=int, default=20)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero for missing rank/raw/kernel/timeline exports")
    args = parser.parse_args()
    root = args.profile_root.resolve()
    if not root.is_dir():
        parser.error("profile_root must be an existing directory")
    dirs = ([path.resolve() for path in args.trace_dir] if args.trace_dir else
            [root] if root.name.endswith("_ascend_pt") else
            sorted(path for path in root.rglob("*_ascend_pt") if path.is_dir()))
    if any(not path.is_dir() or not path.is_relative_to(root) for path in dirs):
        parser.error("trace-dir must be an existing directory within profile_root")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "profile_root": str(root),
              "expected_rank_count": args.expected_ranks, "ranks": [], "errors": [],
              "timing_unit": "us", "critical_path_method": "Requires request markers and dependencies; cumulative time is reported separately"}
    for index, path in enumerate(dirs):
        try:
            report["ranks"].append(analyze_rank(path, root, index, args.max_top))
        except (OSError, ValueError, TypeError) as exc:
            report["errors"].append({"directory": str(path), "error": f"{type(exc).__name__}: {exc}"})
    ranks = report["ranks"]
    ids = [item["rank"] for item in ranks if item["rank"] is not None]
    report["discovered_trace_count"] = len(dirs)
    report["identified_rank_ids"] = sorted(set(ids))
    report["duplicate_rank_ids"] = sorted(rank for rank, count in Counter(ids).items() if count > 1)
    report["passed"] = bool(ranks) and not report["errors"] and all(item["completeness"]["valid"] for item in ranks)
    if args.expected_ranks is not None:
        report["passed"] &= len(ranks) == args.expected_ranks and len(set(ids)) == args.expected_ranks and set(ids) == set(range(args.expected_ranks))
    report["cross_rank_alignment"] = {
        "status": "Clock-aligned comparison requires profiler metadata establishing a common clock; rank spans retained separately",
        "rank_windows": [{"rank": item["rank"], "window_us": item["kernel_window_us"]} for item in ranks],
    }
    if args.observations_root:
        report["cpu_observations"] = cpu_observations(args.observations_root.resolve())
        report["passed"] &= report["cpu_observations"]["passed"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    operations = [dict(row, rank=item["rank"], trace_directory=item["directory"]) for item in ranks for row in item["all_operations"]]
    fields = ["rank", "name", "op_type", "shape", "dtype", "count", "cumulative_us", "mean_us", "max_us", "interval_union_us", "capture_window_coverage_fraction", "critical_path_contribution_us", "communication", "evidence", "trace_directory"]
    write_csv(args.output_dir / "operations.csv", operations, fields)
    write_csv(args.output_dir / "top20.csv", [dict(row, rank=item["rank"], trace_directory=item["directory"]) for item in ranks for row in item["top_operations"]], fields)
    write_csv(args.output_dir / "communication.csv", [row for row in operations if row["communication"]], fields)
    write_csv(args.output_dir / "cpu_scopes.csv", [dict(row, rank=item["rank"]) for item in ranks for row in item["cpu_scopes"]], fields[:-1])
    if "cpu_observations" in report:
        write_csv(args.output_dir / "cpu_observations.csv", report["cpu_observations"]["stages"],
                  ["name", "pid", "rank", "count", "inclusive_cumulative_ms", "mean_ms", "interval_union_ms", "overlap_with_same_name_ms", "wall_range_ns", "evidence"])
    print(json.dumps({"passed": report["passed"], "trace_count": len(ranks), "rank_ids": report["identified_rank_ids"], "output": str(args.output_dir.resolve()), "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["passed"] or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
