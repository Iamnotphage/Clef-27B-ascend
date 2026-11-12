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
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import re
import sys


COMM = re.compile(r"hccl|hcom|all[_ ]?reduce|all[_ ]?gather|reduce[_ ]?scatter|all[_ ]?to[_ ]?all|(?:^|[/:_ ])(?:broadcast|send|recv)(?:$|[/:_ ])", re.I)
MARKER = re.compile(r"clef|systemone|joint|scheduler|tokeniz|process_input|encode_request|model_forward|execute_model|pooler", re.I)
SYNC = re.compile(r"event[ _]?wait|event[ _]?record|notify[ _]?wait|notify[ _]?record|stream[ _]?wait", re.I)


@lru_cache(maxsize=2048)
def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def finite(value):
    try:
        number = float(str(value).replace(",", ""))
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=128)
def normalized_headers(headers):
    return {norm(key): key for key in headers}


def pick(row, *names):
    indexed = normalized_headers(tuple(row))
    for name in names:
        value = row.get(indexed.get(norm(name)))
        if value not in (None, "", "N/A", "NA", "None"):
            return value
    return None


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, number_strings=False):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream, parse_float=str) if number_strings else json.load(stream)


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
    total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def microseconds_to_ns(value):
    try:
        number = Decimal(str(value).strip().replace(",", ""))
        return int(number * 1000) if number.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def interval(row):
    start = microseconds_to_ns(pick(row, "Start Time(us)", "Task Start Time(us)", "Start Timestamp(us)", "ts"))
    duration = microseconds_to_ns(pick(row, "Duration(us)", "Task Duration(us)", "dur"))
    return (start, start + duration) if start is not None and duration is not None and duration >= 0 else None


def time_fields(times):
    start, end = times
    return {"start_ns": start, "end_ns": end, "duration_ns": end - start,
            "start_us": start / 1000, "end_us": end / 1000, "duration_us": (end - start) / 1000}


def intervals_ns(rows):
    return [(row["start_ns"], row["end_ns"]) for row in rows]


def computation(row):
    return not row["communication"] and not row.get("synchronization", False)


def synchronization_wait(row):
    return row.get("synchronization", False) and "wait" in (row["name"] + " " + row.get("op_type", "")).lower()


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
                **time_fields(times), "communication": bool(COMM.search(name + " " + op_type)),
                "synchronization": bool(SYNC.search(name + " " + op_type)),
                "shape": pick(row, "Input Shapes", "Input Shape", "Shapes"),
                "dtype": pick(row, "Input Data Types", "Input Data Type", "Input Dtypes", "Dtype"),
                "output_shape": pick(row, "Output Shapes", "Output Shape"),
                "accelerator_core": pick(row, "Accelerator Core", "Task Type"),
                "hardware_metrics": {key: finite(value) for key, value in row.items()
                                     if re.match(r"^(aic|aiv)_|^aicore_time|^cube_utilization", str(key)) and finite(value) is not None},
                "stream": pick(row, "Stream Id", "Stream ID", "Stream"),
                "device": pick(row, "Device_id", "Device ID", "Device Id"),
                "task_id": pick(row, "Task Id", "Task ID"),
                "evidence_file": str(path.relative_to(root)), "evidence_line": line,
            })
    if not rows:
        warnings.append(f"No timed kernels parsed from {path.name}; headers={fields}")
    return rows, fields, warnings


def event_duration(event):
    return interval(event)


def export_lookup(rows):
    exact, task = defaultdict(list), defaultdict(list)
    for row in rows:
        exact[(row["name"], row["start_ns"], row["duration_ns"])].append(row)
        if row.get("task_id") is not None and row.get("stream") is not None:
            task[(row["name"], str(row["task_id"]), str(row["stream"]))].append(row)
    return exact, task


def match_export_row(row, lookup):
    exact, tasks = lookup
    candidates = exact.get((row["name"], row["start_ns"], row["duration_ns"]), [])
    if len(candidates) == 1:
        return candidates[0], "exact name/start/duration", 0
    key = (row["name"], str(row.get("task_id")), str(row.get("stream")))
    candidates = [candidate for candidate in tasks.get(key, [])
                  if abs(candidate["start_ns"] - row["start_ns"]) <= 1000 and
                  abs(candidate["duration_ns"] - row["duration_ns"]) <= 1000]
    if len(candidates) == 1:
        candidate = candidates[0]
        error = max(abs(candidate["start_ns"] - row["start_ns"]), abs(candidate["duration_ns"] - row["duration_ns"]))
        return candidate, "exact name/task/stream with <=1us export rounding check", error
    return None, None, None


def trace_rows(path, root, csv_rows=()):
    data = read_json(path, number_strings=True)
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
            **time_fields(times),
            "communication": bool(COMM.search(name + " " + category)),
            "synchronization": bool(SYNC.search(name + " " + category)),
            "shape": pick(args, "Input Dims", "Input Shapes", "Input Shape"),
            "dtype": pick(args, "Input type", "Input Data Types", "Input Dtypes"),
            "stream": pick(args, "Physic Stream Id", "Stream Id") or event.get("tid"), "device": process,
            "task_id": pick(args, "Task Id", "Task ID"),
            "pid": event.get("pid"), "tid": event.get("tid"),
            "evidence_file": str(path.relative_to(root)), "evidence_event_index": index,
        }
        timed.append(row)
        # Metadata is essential: CPU-side hccl launches are not device kernels.
        on_device = bool(re.search(r"kernel|ascend.?hardware|npu.?hardware", category + " " + process, re.I) or
                         (process.lower() == "communication" and COMM.search(name)))
        if on_device and not re.search(r"runtime|cpu_op|python|torch", category, re.I):
            kernels.append(row)
        elif MARKER.search(name):
            scopes.append(dict(row, pid=event.get("pid"), tid=event.get("tid"), category=category))
    lookup = export_lookup(csv_rows)
    for kernel in kernels:
        matched, method, error = match_export_row(kernel, lookup)
        if matched is not None:
            for key in ("shape", "dtype", "output_shape", "accelerator_core", "hardware_metrics"):
                if key in matched:
                    kernel[key] = matched[key]
            kernel["csv_evidence"] = {"file": matched["evidence_file"], "line": matched["evidence_line"],
                                      "method": method, "max_timestamp_or_duration_error_ns": error}
    flow_summary = correlate_flow_scopes(events, kernels, scopes)
    return kernels, scopes, {
        "event_count": len(events), "timed_event_count": len(timed),
        "device_kernel_event_count": len(kernels),
        "categories": dict(categories), "process_names": process_names,
        "timed_event_range_us": [min(r["start_us"] for r in timed), max(r["end_us"] for r in timed)] if timed else None,
        "timed_event_range_ns": [min(r["start_ns"] for r in timed), max(r["end_ns"] for r in timed)] if timed else None,
        "module_flow_correlation": flow_summary,
    }


def correlate_flow_scopes(events, kernels, scopes):
    """Follow explicit trace flow endpoints from host scopes to device events.

    Module totals are inclusive across nested scopes. This establishes observed
    kernel provenance, while request critical paths still require dependencies.
    """
    per_thread_scopes, per_stream_kernels = defaultdict(list), defaultdict(list)
    for scope in scopes:
        per_thread_scopes[(str(scope.get("pid")), str(scope.get("tid")))].append(scope)
    for kernel in kernels:
        per_stream_kernels[(str(kernel.get("pid")), str(kernel.get("tid")))].append(kernel)
    stream_starts = {}
    for key, group in per_stream_kernels.items():
        group.sort(key=lambda row: row["start_ns"])
        stream_starts[key] = [row["start_ns"] for row in group]
    opened, module_kernels, instance_kernels, instance_scopes = {}, defaultdict(dict), defaultdict(dict), {}
    flow_count = paired_count = assigned_count = 0
    evidence = []
    endpoints = [(index, event) for index, event in enumerate(events) if event.get("ph") in ("s", "f")]
    endpoints.sort(key=lambda pair: (microseconds_to_ns(pair[1].get("ts")) or 0, pair[1].get("ph") == "f"))
    for index, event in endpoints:
        identity = event.get("id", event.get("id2"))
        if identity is None:
            continue
        flow_count += 1
        key = (str(event.get("cat", "")), json.dumps(identity, sort_keys=True))
        if event["ph"] == "s":
            opened[key] = (index, event)
            continue
        source_pair = opened.pop(key, None)
        if source_pair is None:
            continue
        paired_count += 1
        source_index, source = source_pair
        source_time, target_time = microseconds_to_ns(source.get("ts")), microseconds_to_ns(event.get("ts"))
        if source_time is None or target_time is None:
            continue
        marker_group = per_thread_scopes.get((str(source.get("pid")), str(source.get("tid"))), [])
        ancestors = [scope for scope in marker_group if scope["start_ns"] <= source_time <= scope["end_ns"]]
        stream_key = (str(event.get("pid")), str(event.get("tid")))
        target_group = per_stream_kernels.get(stream_key, [])
        target_index = bisect_right(stream_starts.get(stream_key, []), target_time) - 1
        if not ancestors or target_index < 0:
            continue
        # Device tasks in one stream are ordered; a flow target falls in its
        # actual kernel. No approximate matching across other streams is used.
        kernel = target_group[target_index]
        if not kernel["start_ns"] <= target_time <= kernel["end_ns"]:
            continue
        assigned_count += 1
        names = sorted({scope["name"] for scope in ancestors})
        kernel["flow_observed_modules"] = sorted(set(kernel.get("flow_observed_modules", [])) | set(names))
        kernel["flow_scope_event_indices"] = sorted(set(kernel.get("flow_scope_event_indices", [])) |
                                                    {scope["evidence_event_index"] for scope in ancestors})
        kernel["flow_scope_keys"] = sorted(set(kernel.get("flow_scope_keys", [])) |
                                           {f"{scope['evidence_file']}#{scope['evidence_event_index']}" for scope in ancestors})
        for name in names:
            module_kernels[name][kernel["evidence_event_index"]] = kernel
        for scope in ancestors:
            scope_index = scope["evidence_event_index"]
            instance_kernels[scope_index][kernel["evidence_event_index"]] = kernel
            instance_scopes[scope_index] = scope
        if len(evidence) < 30:
            evidence.append({"flow_key": list(key), "source_event_index": source_index,
                             "target_event_index": index, "kernel_event_index": kernel["evidence_event_index"],
                             "source_ts_us": source_time / 1000, "target_ts_us": target_time / 1000,
                             "source_ts_ns": source_time, "target_ts_ns": target_time,
                             "modules": names, "kernel": kernel["name"]})
    summaries = []
    for name, grouped in sorted(module_kernels.items()):
        group = list(grouped.values())
        intervals = intervals_ns(group)
        compute = intervals_ns(row for row in group if computation(row))
        communication = intervals_ns(row for row in group if row["communication"])
        synchronization = intervals_ns(row for row in group if row.get("synchronization", False))
        waits = intervals_ns(row for row in group if synchronization_wait(row))
        active = intervals_ns(row for row in group if not row.get("synchronization", False))
        summaries.append({"module": name, "kernel_count": len(group),
                          "kernel_cumulative_us": sum(row["duration_ns"] for row in group) / 1000,
                          "kernel_union_us": union_us(intervals) / 1000, "compute_union_us": union_us(compute) / 1000,
                          "compute_kernel_union_us": union_us(compute) / 1000,
                          "active_kernel_union_us": union_us(active) / 1000,
                          "synchronization_union_us": union_us(synchronization) / 1000,
                          "sync_wait_union_us": union_us(waits) / 1000,
                          "communication_union_us": union_us(communication) / 1000,
                          "compute_communication_overlap_us": intersection_us(compute, communication) / 1000,
                          "device_range_us": [min(a for a, _ in intervals) / 1000, max(b for _, b in intervals) / 1000],
                          "critical_path_contribution_us": None,
                          "operations": summarize_rows(group)})
    instances = []
    for scope_index, grouped in sorted(instance_kernels.items()):
        scope = instance_scopes[scope_index]
        group = list(grouped.values())
        intervals = intervals_ns(group)
        communication = intervals_ns(row for row in group if row["communication"])
        compute = intervals_ns(row for row in group if computation(row))
        synchronization = intervals_ns(row for row in group if row.get("synchronization", False))
        waits = intervals_ns(row for row in group if synchronization_wait(row))
        active = intervals_ns(row for row in group if not row.get("synchronization", False))
        device_range = [min(a for a, _ in intervals), max(b for _, b in intervals)]
        instances.append({"module": scope["name"], "scope_event_index": scope_index,
                          "cpu_start_us": scope["start_us"], "cpu_end_us": scope["end_us"],
                          "cpu_start_ns": scope["start_ns"], "cpu_end_ns": scope["end_ns"],
                          "cpu_inclusive_duration_us": scope["duration_us"],
                          "device_start_us": device_range[0] / 1000, "device_end_us": device_range[1] / 1000,
                          "device_start_ns": device_range[0], "device_end_ns": device_range[1],
                          "device_span_us": (device_range[1] - device_range[0]) / 1000,
                          "kernel_count": len(group), "kernel_cumulative_us": sum(row["duration_ns"] for row in group) / 1000,
                          "kernel_union_us": union_us(intervals) / 1000, "communication_union_us": union_us(communication) / 1000,
                          "active_kernel_union_us": union_us(active) / 1000,
                          "compute_kernel_union_us": union_us(compute) / 1000,
                          "synchronization_union_us": union_us(synchronization) / 1000,
                          "sync_wait_union_us": union_us(waits) / 1000,
                          "compute_communication_overlap_us": intersection_us(compute, communication) / 1000,
                          "pid": scope["pid"], "tid": scope["tid"]})
    return {"flow_endpoint_count": flow_count, "paired_flow_count": paired_count,
            "flow_pairs_assigned_to_modules": assigned_count, "modules": summaries,
            "module_instances": instances, "evidence": evidence,
            "method": "Explicit flow s/f endpoints: source lies within host marker and target within device kernel on its exact pid/tid",
            "scope": "Module kernel totals include nested modules; totals across modules overlap and remain separate from request critical paths"}


def canonical_name(name):
    # HCCL embeds per-invocation sequence identifiers in otherwise identical
    # operation names. Group the invocations while retaining exact evidence.
    return re.sub(r"^(hcom_[A-Za-z]+)__?\d+_\d+_\d+(?:@.*)?$", r"\1", name)


def unique_group_coverage(grouped, predicate=lambda row: True):
    boundaries = [(value, index, delta) for index, group in enumerate(grouped.values()) for row in group
                  for value, delta in ((row["start_ns"], 1), (row["end_ns"], -1))
                  if row["end_ns"] > row["start_ns"] and predicate(row)]
    active, unique = {}, Counter()
    previous = None
    for when, updates in groupby(sorted(boundaries), key=lambda value: value[0]):
        if previous is not None and len(active) == 1:
            unique[next(iter(active))] += when - previous
        for _, index, delta in updates:
            count = active.get(index, 0) + delta
            if count:
                active[index] = count
            else:
                active.pop(index, None)
        previous = when
    return unique


def summarize_rows(rows, capture_span=None):
    grouped = defaultdict(list)
    for row in rows:
        key = (canonical_name(row["name"]), row.get("op_type", ""), json.dumps(row.get("shape"), sort_keys=True), json.dumps(row.get("dtype"), sort_keys=True))
        grouped[key].append(row)
    # Sweep actual device intervals once. A group's unique coverage is time
    # during which that group is active and every other operation group is
    # inactive. This is occupancy evidence, independent of DAG critical paths.
    unique = unique_group_coverage(grouped)
    unique_compute = unique_group_coverage(grouped, computation)
    compute_total_union = union_us(intervals_ns(row for group in grouped.values() for row in group if computation(row))) / 1000
    result = []
    for index, ((name, op_type, shape, dtype), group) in enumerate(grouped.items()):
        times = intervals_ns(group)
        cumulative = sum(row["duration_ns"] for row in group) / 1000
        covered = union_us(times) / 1000
        host_scope = all("category" in row for row in group)
        compute_covered = union_us(intervals_ns(row for row in group if computation(row))) / 1000
        metrics = defaultdict(list)
        for row in group:
            for key, value in row.get("hardware_metrics", {}).items():
                metrics[key].append((value, row["duration_ns"]))
        metric_summary = {}
        for key, values in sorted(metrics.items()):
            duration = sum(weight for _, weight in values)
            metric_summary[key] = {"samples": len(values), "min": min(value for value, _ in values),
                                   "max": max(value for value, _ in values),
                                   "duration_weighted_mean": sum(value * weight for value, weight in values) / duration if duration else None}
            if "time(us)" in key:
                metric_summary[key]["sum_us"] = sum(value for value, _ in values)
        result.append({
            "name": name, "op_type": op_type, "shape": json.loads(shape), "dtype": json.loads(dtype),
            "count": len(group), "cumulative_us": cumulative, "mean_us": cumulative / len(group),
            "max_us": max(row["duration_us"] for row in group), "interval_union_us": covered,
            "nonoverlapped_device_coverage_us": None if host_scope else unique[index] / 1000,
            "active_compute_unique_coverage_us": None if host_scope else unique_compute[index] / 1000,
            "active_compute_time_removal_upper_bound_us": None if host_scope else compute_covered,
            "active_compute_union_denominator_us": None if host_scope else compute_total_union,
            "active_compute_coverage_fraction": compute_covered / compute_total_union if compute_total_union and not host_scope else None,
            "nonoverlapped_host_scope_coverage_us": unique[index] / 1000 if host_scope else None,
            "unique_coverage_scope": "Other operation groups within this summarized collection; module subsets exclude kernels outside that module",
            "device_time_removal_upper_bound_us": None if host_scope else covered,
            "capture_window_coverage_fraction": covered / capture_span if capture_span else None,
            "critical_path_contribution_us": None,
            "communication": any(row["communication"] for row in group),
            "synchronization": any(row.get("synchronization", False) for row in group),
            "observed_names_sample": list(dict.fromkeys(row["name"] for row in group))[:3],
            "accelerator_cores": sorted({row["accelerator_core"] for row in group if row.get("accelerator_core")}),
            "hardware_metrics": metric_summary,
            "flow_observed_modules": sorted({name for row in group for name in row.get("flow_observed_modules", [])}),
            "evidence": [{key: row[key] for key in ("name", "evidence_file", "evidence_line", "evidence_event_index", "start_us", "end_us", "start_ns", "end_ns", "stream") if key in row} for row in group[:3]],
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
            name = keys[-1] if keys else ""
            collective = next((kind for pattern, kind in (
                ("allreduce", "all_reduce"), ("allgather", "all_gather"),
                ("reducescatter", "reduce_scatter"), ("alltoall", "all_to_all"),
                ("broadcast", "broadcast"), ("receive", "receive"), ("send", "send"))
                               if pattern in norm(name)), "other")
            result.append({"operation_path": "/".join(keys), "time_info": time_info, "bandwidth_info": bandwidth,
                           "aggregate": norm(name).startswith("total"), "collective_type": collective,
                           "reported_elapsed_us": microseconds(time_info, "Elapse Time(ms)", "Elapsed Time(ms)"),
                           "reported_wait_us": microseconds(time_info, "Wait Time(ms)"),
                           "reported_synchronization_us": microseconds(time_info, "Synchronization Time(ms)"),
                           "evidence_file": str(path.relative_to(root)), "evidence_json_path": keys})
            return
        for key, item in value.items():
            visit(item, keys + [key])

    visit(read_json(path), [])
    return result


def communication_summary(details):
    groups = defaultdict(list)
    for row in details:
        if not row["aggregate"]:
            groups[row["collective_type"]].append(row)
    result = []
    for collective, rows in sorted(groups.items()):
        item = {"collective_type": collective, "operation_count": len(rows)}
        for field in ("reported_elapsed_us", "reported_wait_us", "reported_synchronization_us"):
            values = [row[field] for row in rows if row[field] is not None]
            item[field + "_sum"] = sum(values) if values else None
            item[field + "_samples"] = len(values)
        item["evidence_operation_paths"] = [row["operation_path"] for row in rows[:3]]
        result.append(item)
    return {"by_collective": result, "aggregate_entries_excluded": sum(row["aggregate"] for row in details),
            "method": "Counts distinct non-aggregate communication.json entries; HCCL phase/kernel counts are reported separately. Elapsed sums can overlap and are not critical-path duration"}


def profiler_metadata(path):
    result = []
    for info in sorted(path.glob("profiler_info*.json")):
        try:
            data = read_json(info)
            result.append({"path": str(info.resolve()), "sha256": file_sha(info), "data": data})
        except (OSError, ValueError):
            pass
    return result


def rank_identity(path, metadata):
    for info in metadata:
        queue = [(info["data"], [])]
        while queue:
            value, keys = queue.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if norm(key) in ("rankid", "rank") and isinstance(item, (int, str)) and re.fullmatch(r"\d+", str(item)):
                        return {"rank": int(item), "source": "profiler metadata", "file": info["path"], "json_path": keys + [key]}
                    if isinstance(item, dict):
                        queue.append((item, keys + [key]))
    return {"rank": None, "source": "unidentified; raw directory names alone do not establish rank"}


def memory_summary(directory, root):
    """Keep exported memory components/devices separate, including WORKSPACE."""
    groups, allocations, files = defaultdict(list), [], []
    for path in sorted(directory.glob("*memory*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            files.append({"path": str(path.relative_to(root)), "headers": fields})
            for line, row in enumerate(reader, 2):
                evidence = {"file": str(path.relative_to(root)), "line": line}
                allocated = finite(pick(row, "Total Allocated(MB)"))
                reserved = finite(pick(row, "Total Reserved(MB)"))
                active = finite(pick(row, "Total Active(MB)"))
                if any(value is not None for value in (allocated, reserved, active)):
                    component, device = pick(row, "Component") or "unknown", pick(row, "Device Type") or "unknown"
                    groups[(component, device)].append({"timestamp_us": finite(pick(row, "Timestamp(us)")),
                                                       "allocated_reported_mb": allocated, "reserved_reported_mb": reserved,
                                                       "active_reported_mb": active, "evidence": evidence})
                size = finite(pick(row, "Size(KB)"))
                if size is not None:
                    allocations.append({"name": pick(row, "Name"), "size_reported_kb": size,
                                        "allocation_time_us": finite(pick(row, "Allocation Time(us)")),
                                        "release_time_us": finite(pick(row, "Release Time(us)")),
                                        "duration_us": finite(pick(row, "Duration(us)")),
                                        "device": pick(row, "Device Type"), "evidence": evidence})
    summaries = []
    for (component, device), rows in sorted(groups.items()):
        peaks = {}
        for kind in ("allocated", "reserved", "active"):
            key = f"{kind}_reported_mb"
            valid = [row for row in rows if row[key] is not None]
            maximum = max(valid, key=lambda row: row[key]) if valid else None
            peaks[f"peak_{key}"] = maximum[key] if maximum else None
            peaks[f"peak_{kind}_evidence"] = maximum["evidence"] if maximum else None
        timestamps = [row["timestamp_us"] for row in rows if row["timestamp_us"] is not None]
        summaries.append(dict(component=component, device=device, record_count=len(rows),
                              timestamp_range_us=[min(timestamps), max(timestamps)] if timestamps else None, **peaks))
    return {"status": "exported" if summaries or allocations else "no memory records exported", "source_files": files,
            "component_device_peaks": summaries, "workspace_component_present": any("workspace" in component.lower() for component, _ in groups),
            "observed_operator_allocation_count": len(allocations),
            "top20_operator_allocations": sorted(allocations, key=lambda row: row["size_reported_kb"], reverse=True)[:20],
            "method": "Peak recorded values per explicit component/device; component totals may overlap. Capture can omit earlier allocations and does not classify weights/cache by itself",
            "units": "MB/KB retained exactly as labelled by exporter; verify exporter byte conversion before unit reinterpretation"}


def analyze_rank(path, root, index, max_top):
    warnings = []
    metadata = profiler_metadata(path)
    identity = rank_identity(path, metadata)
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
            kernels, markers, trace_metadata = trace_rows(trace, root, rows)
            timeline_kernel_rows.extend(kernels)
            scopes.extend(markers)
            timeline_metadata.append(dict(trace_metadata, file=str(trace.relative_to(root))))
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"Cannot parse {trace.name}: {type(exc).__name__}: {exc}")
    kernel_source = "kernel CSV" if rows else "timeline device events"
    if not rows:
        rows = timeline_kernel_rows
    else:
        # Transfer only exact exported event identity matches. Different clocks
        # or rounded timestamps leave module ownership unassigned.
        lookup = export_lookup(timeline_kernel_rows)
        for row in rows:
            matched, method, error = match_export_row(row, lookup)
            if matched is not None:
                row["flow_observed_modules"] = matched.get("flow_observed_modules", [])
                row["flow_scope_keys"] = matched.get("flow_scope_keys", [])
                row["module_correlation_method"] = method
                row["module_correlation_max_time_error_ns"] = error
    all_times = intervals_ns(rows)
    compute_times = intervals_ns(row for row in rows if computation(row))
    comm_times = intervals_ns(row for row in rows if row["communication"])
    sync_times = intervals_ns(row for row in rows if row.get("synchronization", False))
    wait_times = intervals_ns(row for row in rows if synchronization_wait(row))
    window_ns = [min(a for a, _ in all_times), max(b for _, b in all_times)] if all_times else None
    window = [value / 1000 for value in window_ns] if window_ns else None
    span = (window_ns[1] - window_ns[0]) / 1000 if window_ns else None
    all_union, compute_union, comm_union = (union_us(times) / 1000 for times in (all_times, compute_times, comm_times))
    overlap = intersection_us(compute_times, comm_times) / 1000
    operations = summarize_rows(rows, span)
    scoped_rows = defaultdict(list)
    for row in rows:
        for scope_key in row.get("flow_scope_keys", []):
            scoped_rows[scope_key].append(row)
    request_windows = []
    for trace in timeline_metadata:
        for instance in trace["module_flow_correlation"]["module_instances"]:
            if instance["module"] != "clef::worker_execute_model":
                continue
            key = f"{trace['file']}#{instance['scope_event_index']}"
            subset = scoped_rows.get(key, [])
            if not subset:
                continue
            times = intervals_ns(subset)
            compute = intervals_ns(row for row in subset if computation(row))
            communication = intervals_ns(row for row in subset if row["communication"])
            first, last = min(a for a, _ in times), max(b for _, b in times)
            request_windows.append({"scope_key": key, "cpu_start_ns": instance["cpu_start_ns"],
                                    "cpu_end_ns": instance["cpu_end_ns"], "pid": instance["pid"],
                                    "mapped_csv_kernel_count": len(subset), "flow_device_event_count": instance["kernel_count"],
                                    "device_start_ns": first, "device_end_ns": last,
                                    "device_span_us": (last - first) / 1000,
                                    "kernel_union_us": union_us(times) / 1000,
                                    "compute_union_us": union_us(compute) / 1000,
                                    "communication_union_us": union_us(communication) / 1000,
                                    "compute_communication_overlap_us": intersection_us(compute, communication) / 1000,
                                    "top_operations": summarize_rows(subset, (last - first) / 1000)[:max_top],
                                    "method": "CSV tasks matched by exact times or exact name/task/stream with <=1us rounding checks to trace events linked by explicit flows; unmatched tasks remain outside this attribution"})
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
        "rank": identity["rank"], "rank_identity": identity, "profiler_metadata": metadata,
        "discovery_index": index, "directory": str(path.resolve()),
        "relative_directory": str(path.relative_to(root)), "completeness": completeness,
        "raw_file_count": len(files) - len(export_files), "total_bytes": sum(item.stat().st_size for item in files),
        "exported_file_index": [{"path": str(item.relative_to(root)), "bytes": item.stat().st_size, "sha256": file_sha(item)} for item in export_files],
        "kernel_source": kernel_source, "kernel_csv_headers": headers, "shape_status": shape_status,
        "kernel_window_us": window, "kernel_window_ns": window_ns, "kernel_window_span_us": span,
        "kernel_cumulative_us": sum(row["duration_ns"] for row in rows) / 1000,
        "kernel_union_us": all_union, "compute_union_us": compute_union, "communication_union_us": comm_union,
        "compute_kernel_union_us": compute_union,
        "synchronization_union_us": union_us(sync_times) / 1000, "sync_wait_union_us": union_us(wait_times) / 1000,
        "compute_communication_overlap_us": overlap, "communication_nonoverlapped_us": comm_union - overlap,
        "no_device_kernel_within_window_us": span - all_union if span else None,
        "kernel_count": len(rows), "communication_kernel_count": sum(row["communication"] for row in rows),
        "top_operations": operations[:max_top], "all_operations": operations,
        "communication_operations": [row for row in operations if row["communication"]],
        "communication_details": comm_ops, "cpu_scopes": summarize_rows(scopes, span),
        "communication_export_summary": communication_summary(comm_ops),
        "cpu_scope_events": scopes,
        "timeline_metadata": timeline_metadata, "warnings": warnings,
        "request_device_windows": request_windows,
        "memory": memory_summary(search, root),
        "method": {"busy": "union of exported device kernel intervals in the device clock domain",
                   "communication": "case-insensitive HCCL/collective name classification; inspect operation names",
                   "idle": "gaps between first and last exported kernel; includes inter-request idle and host work",
                   "critical_path": "Not inferred from cumulative kernel duration; request markers and event dependencies required",
                   "unique_coverage": "Time a canonical operation/shape/dtype group is active while all other groups on this rank are inactive; device occupancy evidence, separate from DAG critical paths",
                   "active_compute_unique_coverage": "Same interval sweep restricted to computation tasks; synchronization waits/records, HCCL parent ranges and communication AICPU tasks are excluded from competing groups. Denominator is the union of active computation in this collection",
                   "device_time_removal_upper_bound": "Interval union of the group; removing the group's device work can remove at most this occupancy within the capture. Host dependencies and other ranks can limit realized benefit",
                   "shape": "Taken only from explicit exported shapes; no inference from names",
                   "timestamp_precision": "Integer nanosecond interval arithmetic; displayed epoch microsecond floats are approximate, evidence *_ns fields retain precision",
                   "hardware_metrics": "Exported metric min/max and kernel-duration-weighted means; hardware pipeline times remain separate from request critical-path duration"},
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def cpu_observations(root, benchmark_reports=()):
    """Read opt-in host observations; preserve inclusive/overlapping semantics."""
    events, errors, files, selected_benchmarks, cache_observations = [], [], [], [], []
    aliases, client_rows, selected_windows = {}, {}, []

    def wall_ns(value):
        parsed = datetime.fromisoformat(value)
        return int(parsed.timestamp()) * 1000000000 + parsed.microsecond * 1000

    for path in benchmark_reports:
        document = read_json(path)
        selected_benchmarks.append({"path": str(path.resolve()), "sha256": file_sha(path)})
        rows = document.get("requests", [])
        if not isinstance(rows, list):
            raise ValueError(f"benchmark requests must be a list: {path}")
        for row in rows:
            request_id = row.get("client_request_id")
            if not isinstance(request_id, str):
                raise ValueError(f"benchmark measured row requires client_request_id: {path}")
            client_rows[request_id] = row
            for alias in (request_id, "decision-" + request_id, "decision-" + request_id + "-0", row.get("response_request_id")):
                if isinstance(alias, str) and alias:
                    aliases[alias] = request_id
            selected_windows.append((wall_ns(row["started_at"]), wall_ns(row["finished_at"])))
    selected_windows = merge_intervals(selected_windows)
    unselected_count = 0

    def resolve_identity(value):
        value = str(value)
        if value in aliases:
            return aliases[value]
        # Fixed vLLM appends the completion index and an internal eight-hex
        # suffix. Strip only the observed suffix and require a known base ID.
        match = re.fullmatch(r"(.+)-[0-9a-fA-F]{8}", value)
        return aliases.get(match[1]) if match else None
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
                    end = start + duration
                    event = dict(value, end_wall_ns=end, evidence_file=str(path.resolve()), evidence_line=line)
                    if isinstance(value.get("cache_memory"), dict):
                        cache_observations.append({"pid": value.get("pid"), "rank": value.get("rank"),
                                                   "cache_memory": value["cache_memory"],
                                                   "evidence_file": str(path.resolve()), "evidence_line": line})
                    batch_ids = value.get("batch_request_ids") if isinstance(value.get("batch_request_ids"), list) else []
                    event["batch_request_ids"] = batch_ids
                    event["shared_batch"] = len(batch_ids) > 1
                    if benchmark_reports:
                        request_id = value.get("request_id")
                        batch_matches = sorted({matched for item in batch_ids if (matched := resolve_identity(item)) is not None})
                        if request_id is not None:
                            canonical = resolve_identity(request_id)
                            if canonical is None:
                                unselected_count += 1
                                continue
                            event.update(matched_client_request_id=canonical, selection_basis="request ID")
                        elif batch_matches:
                            event.update(matched_client_request_ids=batch_matches, selection_basis="batch membership; duration shared by batch")
                        elif value["name"] == "clef::worker_execute_model" and not batch_ids:
                            unselected_count += 1
                            continue
                        elif not any(start < stop and end > begin for begin, stop in selected_windows):
                            unselected_count += 1
                            continue
                        else:
                            event["selection_basis"] = "host wall overlap; request ID unavailable"
                    events.append(event)
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({"path": str(path.resolve()), "line": line, "error": str(exc)})
    groups, requests = defaultdict(list), defaultdict(list)
    for event in events:
        groups[(event["name"], event.get("pid"), event.get("rank"))].append(event)
        identities = event.get("matched_client_request_ids", [])
        if event.get("request_id") is not None:
            identities = [str(event.get("matched_client_request_id", event["request_id"]))]
        elif not benchmark_reports:
            identities = [str(item) for item in event.get("batch_request_ids", [])]
        for identity in set(identities):
            requests[identity].append(event)

    def summary(group):
        intervals = [(event["start_wall_ns"], event["end_wall_ns"]) for event in group]
        cumulative = sum(event["duration_ns"] for event in group)
        union = union_us(intervals)  # Interval arithmetic retains input units.
        return {"count": len(group), "inclusive_cumulative_ms": cumulative / 1e6,
                "shared_batch_event_count": sum(event.get("shared_batch", False) for event in group),
                "mean_ms": cumulative / len(group) / 1e6, "interval_union_ms": union / 1e6,
                "overlap_with_same_name_ms": (cumulative - union) / 1e6,
                "wall_range_ns": [min(a for a, _ in intervals), max(b for _, b in intervals)],
                "evidence": [{"file": event["evidence_file"], "line": event["evidence_line"]} for event in group[:3]]}

    stages = [dict(name=name, pid=pid, rank=rank, **summary(group)) for (name, pid, rank), group in groups.items()]
    stages.sort(key=lambda row: row["inclusive_cumulative_ms"], reverse=True)
    request_index = [{"request_id": request_id, "event_count": len(group),
                      "observed_request_ids": sorted({str(identity) for event in group
                                                       for identity in ([event["request_id"]] if event.get("request_id") is not None else event.get("batch_request_ids", []))}),
                      "names": dict(Counter(event["name"] for event in group)),
                      "process_ids": sorted({event.get("pid") for event in group if event.get("pid") is not None}),
                      "wall_range_ns": [min(event["start_wall_ns"] for event in group), max(event["end_wall_ns"] for event in group)],
                      "evidence": [{"file": event["evidence_file"], "line": event["evidence_line"]} for event in group]}
                     for request_id, group in sorted(requests.items())]
    for row in request_index:
        group = requests[row["request_id"]]
        client = client_rows.get(row["request_id"])
        if client is not None:
            row["client"] = {key: client.get(key) for key in ("passed", "elapsed_ms", "client_queue_ms", "http_ms", "json_decode_ms", "protocol_validation_ms", "actual_input_tokens", "field_count", "candidate_count")}
        row["native_engine_metrics"] = [event["engine_metrics"] for event in group if isinstance(event.get("engine_metrics"), dict)]
        per_stage = defaultdict(list)
        for event in group:
            per_stage[event["name"]].append(event)
        row["stages"] = {name: summary(stage) for name, stage in sorted(per_stage.items())}
        per_process_stage = defaultdict(list)
        for event in group:
            per_process_stage[(event["name"], event.get("pid"), event.get("rank"))].append(event)
        row["stages_by_process_rank"] = [dict(name=name, pid=pid, rank=rank, **summary(stage))
                                         for (name, pid, rank), stage in per_process_stage.items()]
        row["stage_aggregation_scope"] = "Inclusive ranges can overlap across names/processes/ranks; shared-batch durations belong to the whole batch"
    engine_summaries = {}
    for key in ("queue_seconds", "scheduled_to_output_seconds", "scheduled_to_first_output_seconds"):
        values = sorted(finite(event.get("engine_metrics", {}).get(key)) for event in events
                        if isinstance(event.get("engine_metrics"), dict) and finite(event["engine_metrics"].get(key)) is not None)
        if values:
            def percentile(fraction):
                position = (len(values) - 1) * fraction
                lower, upper = math.floor(position), math.ceil(position)
                return (values[lower] + (values[upper] - values[lower]) * (position - lower)) * 1000
            engine_summaries[key] = {"sample_count": len(values), "sum_ms": sum(values) * 1000,
                                     "mean_ms": sum(values) / len(values) * 1000,
                                     "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
                                     "min_ms": values[0] * 1000, "max_ms": values[-1] * 1000,
                                     "scope": "Native differences of same-engine-process monotonic timestamps; no direct cross-process timestamp alignment"}
    return {"root": str(root.resolve()), "passed": bool(events) and not errors,
            "source_files": files, "event_count": len(events),
            "selected_benchmark_reports": selected_benchmarks, "excluded_event_count": unselected_count,
            "selection_wall_windows_ns": selected_windows,
            "errors": errors, "stages": stages, "requests": request_index, "events": events,
            "native_engine_metric_summaries": engine_summaries,
            "cache_pool_observations": cache_observations,
            "method": {"timing_scope": "host_wall_inclusive; nested stages and concurrent tasks overlap",
                       "end_wall_ns": "start_wall_ns plus monotonic duration_ns; assumes stable wall clock within each scope",
                       "union": "computed per name/pid/rank; stage unions are not additive across names",
                       "device_alignment": "Host wall nanoseconds are retained separately from exported NPU microseconds; clock calibration is required for alignment"}}


def compact_evidence(report, output_dir):
    """Keep reviewable case evidence small; full records remain in summary/CSV."""
    keys = ("rank", "rank_identity", "profiler_metadata", "directory", "relative_directory", "completeness",
            "raw_file_count", "total_bytes", "exported_file_index", "kernel_source", "shape_status",
            "kernel_window_ns", "kernel_window_span_us", "kernel_cumulative_us", "kernel_union_us",
            "compute_kernel_union_us", "communication_union_us", "compute_communication_overlap_us",
            "synchronization_union_us", "sync_wait_union_us", "communication_nonoverlapped_us",
            "no_device_kernel_within_window_us", "kernel_count", "communication_kernel_count", "warnings")
    operation_keys = ("name", "op_type", "shape", "dtype", "count", "cumulative_us", "mean_us",
                      "interval_union_us", "nonoverlapped_device_coverage_us", "active_compute_unique_coverage_us",
                      "active_compute_time_removal_upper_bound_us", "active_compute_union_denominator_us",
                      "active_compute_coverage_fraction", "device_time_removal_upper_bound_us",
                      "critical_path_contribution_us", "communication", "synchronization", "flow_observed_modules")
    pipe_keys = ("aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio", "aic_icache_miss_rate",
                 "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_icache_miss_rate", "cube_utilization(%)")
    result = {key: value for key, value in report.items() if key not in ("ranks", "cpu_observations")}
    result["compact_authoring_script"] = {"path": str(Path(__file__).resolve()), "sha256": file_sha(Path(__file__))}
    result["ranks"] = []
    result["methods"] = {"timing": "Integer ns arithmetic; exported durations in us. Kernel sums, interval unions and host inclusive ranges remain distinct",
                         "active_compute_unique": "Only computation tasks compete for active-only unique coverage. Sync waits/records, HCCL parent ranges and communication AICPU are excluded; denominator is active compute union in the same collection",
                         "modules": "Means over request-associated marker instances; parent and child scopes overlap. Head active compute and upstream EVENT WAIT are separate",
                         "critical_path": "Explicit flow proves provenance. Unique coverage/removal upper bounds measure occupancy; full DAG critical-path improvement requires a follow-up experiment",
                         "pipe_metrics": "Kernel-duration-weighted reported ratios (0..1), plus exporter cube_utilization(%); full min/max/samples in remote summary"}

    def selected_transport(row):
        bandwidth = row.get("bandwidth_info", {})
        for transport in ("SIO", "HCCS", "RDMA", "PCIE", "SDMA", "UB"):
            item = bandwidth.get(transport, {})
            size = finite(item.get("Transit Size(MB)"))
            if size is not None and size > 0:
                return transport, round(size * 1024**2), item
        return None, None, {}

    for rank in report["ranks"]:
        item = {key: rank[key] for key in keys}
        item["request_device_window_count"] = len(rank["request_device_windows"])
        item["top_operations"] = []
        for operation in rank["top_operations"]:
            compact = {key: operation.get(key) for key in operation_keys}
            compact["hardware_metrics"] = {key: {"duration_weighted_mean": round(value["duration_weighted_mean"], 6) if value["duration_weighted_mean"] is not None else None}
                                           for key, value in operation.get("hardware_metrics", {}).items() if key in pipe_keys}
            compact["evidence"] = operation["evidence"][:1]
            item["top_operations"].append(compact)
        details = [row for row in rank["communication_details"] if not row["aggregate"]]
        item["communication_export_summary"] = rank["communication_export_summary"]
        sizes = {}
        for row in details:
            transport, size, bandwidth = selected_transport(row)
            key = (row["collective_type"], transport, size)
            bucket = sizes.setdefault(key, {"collective_type": key[0], "transport": transport,
                                            "reported_transit_bytes": size, "operation_count": 0,
                                            "transit_ms_sum": 0.0, "elapsed_us_sum": 0.0})
            bucket["operation_count"] += 1
            bucket["transit_ms_sum"] += finite(bandwidth.get("Transit Time(ms)")) or 0
            bucket["elapsed_us_sum"] += row["reported_elapsed_us"] or 0
        item["communication_size_distribution"] = list(sizes.values())
        item["communication_size_scope"] = "One preferred transport per call (SIO first); MB converted with 1024**2. SIO/SDMA aliases are not added. Values describe reported transit bytes; tensor payload interpretation depends on collective algorithm"
        item["communication_examples"] = []
        for row in sorted(details, key=lambda value: value["reported_elapsed_us"] or 0, reverse=True)[:3]:
            transport, size, _ = selected_transport(row)
            item["communication_examples"].append({key: row[key] for key in ("operation_path", "collective_type", "reported_elapsed_us", "reported_wait_us", "reported_synchronization_us", "evidence_file", "evidence_json_path")} |
                                                    {"transport": transport, "reported_transit_bytes": size})
        item["timeline_metadata"], item["module_averages"] = [], []
        for trace in rank["timeline_metadata"]:
            correlation = trace["module_flow_correlation"]
            metadata = {key: value for key, value in trace.items() if key != "module_flow_correlation"}
            metadata["module_flow_correlation"] = {key: correlation[key] for key in ("flow_endpoint_count", "paired_flow_count", "flow_pairs_assigned_to_modules", "method", "scope")}
            metadata["module_flow_correlation"]["evidence"] = correlation["evidence"][:3]
            item["timeline_metadata"].append(metadata)
            groups = defaultdict(list)
            for instance in correlation["module_instances"]:
                if instance.get("request_ids") or "cpu_observations" not in report:
                    groups[instance["module"]].append(instance)
            for name, group in sorted(groups.items()):
                metrics = ("cpu_inclusive_duration_us", "device_span_us", "compute_kernel_union_us", "communication_union_us", "sync_wait_union_us", "active_kernel_union_us")
                means = {key.removesuffix("_us") + "_ms_mean": round(sum(row[key] for row in group) / len(group) / 1000, 6) for key in metrics}
                ids = {identity for row in group for identity in row.get("request_ids", [])}
                item["module_averages"].append({"module": name, "instances": len(group), "matched_request_count": len(ids),
                                                "metrics_per_instance": means,
                                                "evidence": [{"trace_file": trace["file"], "scope_event_index": row["scope_event_index"], "request_ids": row.get("request_ids", [])} for row in group[:3]]})
        memory = rank["memory"]
        item["memory"] = {key: value for key, value in memory.items() if key != "top20_operator_allocations"}
        item["memory"]["top3_operator_allocations"] = memory["top20_operator_allocations"][:3]
        result["ranks"].append(item)
    if "cpu_observations" in report:
        observation = report["cpu_observations"]
        result["cpu_observations"] = {key: value for key, value in observation.items() if key not in ("events", "requests", "cache_pool_observations", "stages")}
        count = len(observation["requests"])
        result["cpu_observations"]["matched_request_count"] = count
        result["cpu_observations"]["stages"] = [{key: row[key] for key in ("name", "pid", "rank", "count", "mean_ms")} |
                                                 {"per_request_inclusive_ms": row["inclusive_cumulative_ms"] / count if count else None,
                                                  "evidence": row["evidence"][:1]} for row in observation["stages"]]
        result["cpu_observations"]["request_paths_file"] = str((output_dir / "request_paths.csv").resolve())
        result["cpu_observations"]["cache_pool_observations"] = [{key: row.get(key) for key in ("pid", "rank", "evidence_file", "evidence_line")} |
                                                                  {"cache_memory": {key: row["cache_memory"].get(key) for key in ("num_blocks", "layout", "unique_storage_bytes", "unique_storage_bytes_by_kind", "deduplication")}}
                                                                 for row in observation["cache_pool_observations"]]
    result["analysis_file_index"] = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha(path)}
                                     for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "compact-evidence.json"]
    return result


def cpu_decomposition(observations, benchmark_reports):
    requests, sources = [], []
    for path in benchmark_reports:
        document = read_json(path)
        requests.extend(document.get("requests", []))
        sources.append({"path": str(path.resolve()), "sha256": file_sha(path)})
    count = len(requests)
    phase = "+".join(path.parent.name + "/" + path.stem for path in benchmark_reports)
    common = {"phase": phase, "request_count": count,
              "sample_names": dict(Counter(str(row.get("sample_name")) for row in requests)),
              "input_token_histogram": dict(Counter(str(row.get("actual_input_tokens")) for row in requests)),
              "field_count_histogram": dict(Counter(str(row.get("field_count")) for row in requests)),
              "candidate_count_histogram": dict(Counter(str(row.get("candidate_count")) for row in requests))}
    rows = []
    for stage in observations["stages"]:
        rows.append(dict(common, name=stage["name"], pid=stage["pid"], rank=stage["rank"],
                         calls=stage["count"], mean_per_call_ms=stage["mean_ms"],
                         per_request_ms=stage["inclusive_cumulative_ms"] / count if count else None,
                         interval_union_ms=stage["interval_union_ms"], scope="host inclusive; nested stages overlap",
                         evidence=stage["evidence"]))
    for key in ("elapsed_ms", "http_ms", "client_queue_ms", "json_decode_ms", "protocol_validation_ms", "client_total_ms"):
        values = [finite(row.get(key)) for row in requests if finite(row.get(key)) is not None]
        rows.append(dict(common, name="client." + key, pid=None, rank=None, calls=len(values),
                         mean_per_call_ms=sum(values) / len(values) if values else None,
                         per_request_ms=sum(values) / count if count else None,
                         interval_union_ms=None, scope="client benchmark measured requests; original latency definition retained",
                         evidence=sources))
    for key, summary in observations["native_engine_metric_summaries"].items():
        evidence = [{"file": event["evidence_file"], "line": event["evidence_line"]} for event in observations["events"]
                    if isinstance(event.get("engine_metrics"), dict) and event["engine_metrics"].get(key) is not None][:3]
        rows.append(dict(common, name="native_engine." + key, pid=None, rank=None,
                         calls=summary["sample_count"], mean_per_call_ms=summary["mean_ms"],
                         per_request_ms=summary["sum_ms"] / count if count else None,
                         interval_union_ms=None, scope=summary["scope"], evidence=evidence))
    return dict(common, passed=observations["passed"], rows=rows, sources=sources,
                scope="CPU scopes measure dispatch and any host waits inside each scope. NPU completion/active work/synchronization are separate trace measurements. Nested host stages are not additive")


def associate_request_ids(report, tolerance_ns=1000000):
    """Join exported host scopes to matching same-process observer intervals."""
    observed = defaultdict(list)
    for event in report["cpu_observations"]["events"]:
        observed[(str(event.get("pid")), event["name"])].append(event)
    matched = unmatched = 0
    matched_scopes = {}
    for rank in report["ranks"]:
        for trace in rank["timeline_metadata"]:
            for instance in trace["module_flow_correlation"]["module_instances"]:
                candidates = [event for event in observed.get((str(instance["pid"]), instance["module"]), [])
                              if abs(event["start_wall_ns"] - instance["cpu_start_ns"]) <= tolerance_ns and
                              event["start_wall_ns"] - tolerance_ns <= instance["cpu_start_ns"] and
                              instance["cpu_end_ns"] <= event["end_wall_ns"] + tolerance_ns]
                candidates.sort(key=lambda event: abs(event["start_wall_ns"] - instance["cpu_start_ns"]))
                if not candidates or (len(candidates) > 1 and abs(candidates[0]["start_wall_ns"] - instance["cpu_start_ns"]) == abs(candidates[1]["start_wall_ns"] - instance["cpu_start_ns"])):
                    unmatched += 1
                    continue
                event = candidates[0]
                identities = event.get("matched_client_request_ids", [])
                if event.get("request_id") is not None:
                    identities = [str(event.get("matched_client_request_id", event["request_id"]))]
                elif not identities:
                    identities = [str(item) for item in event.get("batch_request_ids", [])]
                match = {"request_ids": identities, "shared_batch": event.get("shared_batch", False),
                         "observer_start_minus_trace_start_ns": event["start_wall_ns"] - instance["cpu_start_ns"],
                         "observer_duration_minus_trace_duration_ns": event["duration_ns"] - (instance["cpu_end_ns"] - instance["cpu_start_ns"]),
                         "observer_evidence": {"file": event["evidence_file"], "line": event["evidence_line"]}}
                instance.update(match)
                matched_scopes[f"{trace['file']}#{instance['scope_event_index']}"] = match
                matched += 1
        for window in rank["request_device_windows"]:
            if window["scope_key"] in matched_scopes:
                window.update(matched_scopes[window["scope_key"]])
    report["request_association"] = {"matched_module_instances": matched, "unmatched_module_instances": unmatched,
                                     "tolerance_ns": tolerance_ns,
                                     "method": "Nearest same-pid and exact-name observer interval with start/containment checks in exported wall-clock domain; explicit trace flows supply device provenance",
                                     "limitation": "Temporal scope matching requires validated host/export clock calibration. Exact input token IDs and execution order remain independent checks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--trace-dir", type=Path, action="append", help="Select a window's raw rank directories; repeat once per rank")
    parser.add_argument("--observations-root", type=Path, help="Read optional profile_instrumentation.py cpu-<pid>.jsonl files")
    parser.add_argument("--benchmark-report", type=Path, action="append", default=[], help="Select measured request IDs/windows in CPU observations; repeat for multiple rounds")
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
              "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": file_sha(Path(__file__))},
              "analysis_argv": sys.argv,
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
        report["cpu_observations"] = cpu_observations(args.observations_root.resolve(), args.benchmark_report)
        report["passed"] &= report["cpu_observations"]["passed"]
        associate_request_ids(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    operations = [dict(row, rank=item["rank"], trace_directory=item["directory"]) for item in ranks for row in item["all_operations"]]
    fields = ["rank", "name", "op_type", "shape", "dtype", "count", "cumulative_us", "mean_us", "max_us", "interval_union_us", "nonoverlapped_device_coverage_us", "active_compute_unique_coverage_us", "active_compute_time_removal_upper_bound_us", "active_compute_union_denominator_us", "active_compute_coverage_fraction", "device_time_removal_upper_bound_us", "capture_window_coverage_fraction", "critical_path_contribution_us", "communication", "synchronization", "unique_coverage_scope", "flow_observed_modules", "accelerator_cores", "hardware_metrics", "observed_names_sample", "evidence", "trace_directory"]
    write_csv(args.output_dir / "operations.csv", operations, fields)
    write_csv(args.output_dir / "top20.csv", [dict(row, rank=item["rank"], trace_directory=item["directory"]) for item in ranks for row in item["top_operations"]], fields)
    write_csv(args.output_dir / "communication.csv", [row for row in operations if row["communication"]], fields)
    write_csv(args.output_dir / "cpu_scopes.csv", [dict(row, rank=item["rank"]) for item in ranks for row in item["cpu_scopes"]], fields[:-1])
    write_csv(args.output_dir / "module_instances.csv", [dict(row, rank=rank["rank"], trace_file=trace["file"])
              for rank in ranks for trace in rank["timeline_metadata"]
              for row in trace["module_flow_correlation"]["module_instances"]],
              ["rank", "module", "scope_event_index", "cpu_start_us", "cpu_end_us", "cpu_inclusive_duration_us",
               "device_start_us", "device_end_us", "device_span_us", "kernel_count", "kernel_cumulative_us",
               "kernel_union_us", "active_kernel_union_us", "compute_kernel_union_us", "synchronization_union_us", "sync_wait_union_us", "communication_union_us", "compute_communication_overlap_us", "pid", "tid", "request_ids", "shared_batch", "observer_start_minus_trace_start_ns", "observer_duration_minus_trace_duration_ns", "observer_evidence", "trace_file"])
    write_csv(args.output_dir / "request_device_windows.csv", [dict(row, rank=rank["rank"]) for rank in ranks for row in rank["request_device_windows"]],
              ["rank", "scope_key", "request_ids", "shared_batch", "cpu_start_ns", "cpu_end_ns", "pid", "device_start_ns", "device_end_ns", "device_span_us", "kernel_union_us", "compute_union_us", "communication_union_us", "compute_communication_overlap_us", "mapped_csv_kernel_count", "flow_device_event_count", "top_operations", "method"])
    if "cpu_observations" in report:
        write_csv(args.output_dir / "cpu_observations.csv", report["cpu_observations"]["stages"],
                  ["name", "pid", "rank", "count", "inclusive_cumulative_ms", "mean_ms", "interval_union_ms", "overlap_with_same_name_ms", "wall_range_ns", "evidence"])
        write_csv(args.output_dir / "request_paths.csv", report["cpu_observations"]["requests"],
                  ["request_id", "observed_request_ids", "event_count", "names", "process_ids", "wall_range_ns", "client", "native_engine_metrics", "stages", "stages_by_process_rank", "stage_aggregation_scope", "evidence"])
        if args.benchmark_report:
            decomposition = cpu_decomposition(report["cpu_observations"], args.benchmark_report)
            (args.output_dir / "cpu-decomposition.json").write_text(json.dumps(decomposition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            write_csv(args.output_dir / "cpu-decomposition.csv", decomposition["rows"],
                      ["phase", "request_count", "sample_names", "input_token_histogram", "field_count_histogram", "candidate_count_histogram", "name", "pid", "rank", "calls", "mean_per_call_ms", "per_request_ms", "interval_union_ms", "scope", "evidence"])
    (args.output_dir / "compact-evidence.json").write_text(json.dumps(compact_evidence(report, args.output_dir), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "trace_count": len(ranks), "rank_ids": report["identified_rank_ids"], "output": str(args.output_dir.resolve()), "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["passed"] or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
