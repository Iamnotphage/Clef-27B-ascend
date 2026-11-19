#!/usr/bin/env python3
"""CPU-only preparation for a fresh H11 sample-based msprof run.

This module never imports torch/torch_npu, never starts msprof, never touches a
device, and never reads a workspace.  It only builds a list argv, validates a
fresh expected task identity, defines an explicitly labelled synthetic sample
record, and classifies mock/synthetic outcomes.  A real msprof output adapter is
a placeholder that returns ``unknown_schema``; real fields are not fabricated.

The existing 8f4b843 supervisor keeps the process-tree, cleanup, settle/gate and
single ``begin_restore`` responsibility.  Its ``--child-argv`` hook receives the
list returned by :func:`build_collector_argv`; no native binding or runner is
changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

MSPROF = "/usr/local/Ascend/cann-9.2.1/tools/profiler/bin/msprof"
MSPROF_VERSION = "26.2.0"
MSPROF_SHA256 = "fe8b7256faea5fcf08b1878744177a368893cd8169ba2fe58957e8fe2dacd01d"
HELP_PACKAGE = "profiling-metadata-msprof-20261120/msprof-help.txt"

DELAY_S = 10
DURATION_S = 120
TOTAL_S = DELAY_S + DURATION_S
OUTER_TIMEOUT_S = 240
TIMEOUT_APPROX_S = 180

# Every value below is fixed by the 33 seq4153 plan and the 31 help table.
# Units are part of the contract: --aic-freq is Hz, never milliseconds.
FLAG_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "--output",
        "value": None,
        "unit": None,
        "help_default": None,
        "range": "directory",
        "help_line": "L9",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "optional, no default; must be a fresh absolute directory",
    },
    {
        "name": "--task-time",
        "value": "l1",
        "unit": None,
        "help_default": "on",
        "range": "l0, l1, l2, l3, on, off",
        "help_line": "L17",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "task profiling data; no unit",
    },
    {
        "name": "--ai-core",
        "value": "on",
        "unit": None,
        "help_default": "on when collecting app profiling",
        "range": "on, off",
        "help_line": "L20",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "AI Core profiling switch",
    },
    {
        "name": "--aic-mode",
        "value": "sample-based",
        "unit": None,
        "help_default": "task-based in AI task mode; sample-based in system mode",
        "range": "task-based, sample-based",
        "help_line": "L21-L24",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "collected in a specific interval",
    },
    {
        "name": "--aic-freq",
        "value": "100",
        "unit": "Hz",
        "help_default": "100 Hz",
        "range": "1-100 Hz",
        "help_line": "L25",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "sampling frequency in hertz, not milliseconds",
    },
    {
        "name": "--aic-metrics",
        "value": "PipeUtilization",
        "unit": None,
        "help_default": "PipeUtilization",
        "range": "ArithmeticUtilization, PipeUtilization, Memory, MemoryL0, ResourceConflictRatio, MemoryUB, L2Cache, MemoryAccess",
        "help_line": "L41-L42",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "default aic metric group",
    },
    {
        "name": "--delay",
        "value": str(DELAY_S),
        "unit": "s",
        "help_default": None,
        "range": "1-4294967295 s",
        "help_line": "L83",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "collect start delay in seconds",
    },
    {
        "name": "--duration",
        "value": str(DURATION_S),
        "unit": "s",
        "help_default": None,
        "range": "1-4294967295 s",
        "help_line": "L84",
        "help_conflict": "none documented",
        "runtime_proven": False,
        "note": "collection duration in seconds; flush/exit/kill/wait semantics unknown",
    },
)

# Order is part of the command contract.
FIXED_ARGS: tuple[tuple[str, str], ...] = (
    ("--task-time", "l1"),
    ("--ai-core", "on"),
    ("--aic-mode", "sample-based"),
    ("--aic-freq", "100"),
    ("--aic-metrics", "PipeUtilization"),
    ("--delay", str(DELAY_S)),
    ("--duration", str(DURATION_S)),
)

FORBIDDEN_COLLECTOR_FLAGS = (
    "--application",  # deprecated; positional app/args form is used
    "--dynamic",
    "--pid",
    "--instr-profiling",
    "--instr-profiling-freq",
)

REQUIRED_TARGET_FIELDS = (
    "kernel_name",
    "stream_id",
    "task_id",
    "raw_submit_ns",
    "raw_end_ns",
    "submit_ns",
    "end_ns",
    "end_kind",
    "raw_clock",
    "clock",
    "offset_ns",
    "clock_provenance",
    "source",
    "provenance",
)

KNOWN_CLOCKS = ("monotonic_ns", "utc_ns")
END_KINDS = ("timeout", "pending_end", "duration_end")

STATUS_CANDIDATE = "capability_candidate"
STATUS_UNPROVEN = "capability_unproven"
STATUS_NOT_SUCCESS = "not_success"

MISSING_REASONS = {
    "missing_kernel_name",
    "missing_stream_id",
    "missing_task_id",
    "missing_raw_submit_ns",
    "missing_raw_end_ns",
    "missing_submit_ns",
    "missing_end_ns",
    "missing_end_kind",
    "missing_raw_clock",
    "missing_clock",
    "missing_offset_ns",
    "missing_clock_provenance",
    "missing_source",
    "missing_provenance",
    "missing_sample_kernel",
    "missing_sample_raw_timestamp",
    "missing_sample_timestamp",
    "missing_sample_stream",
    "missing_sample_task",
    "missing_sample_raw_clock",
    "missing_sample_clock",
    "missing_sample_offset",
    "missing_sample_clock_provenance",
}
HARD_COLLECTOR_REASONS = {
    "collector_timeout",
    "collector_error",
    "output_missing",
    "truncated",
    "collector_exited_early",
    "ranks_alive",
    "unknown_schema",
    "timeline_only",
    "sample_empty",
    "synthetic_not_labeled",
    "invalid_synthetic_record",
    "multiple_candidates",
    "ambiguous_target",
    "clock_mismatch",
    "unknown_clock",
    "unknown_end_kind",
    "invalid_window",
    "sample_invalid",
    "conversion_mismatch",
    "identity_offset_nonzero",
}


class ProfileError(ValueError):
    """Invalid argv/plan/record input; never silently repaired."""


def flag_specs() -> list[dict[str, Any]]:
    """Return a JSON-serializable copy of the fixed flag table."""
    return [dict(item) for item in FLAG_SPECS]


def _string_list(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        raise ProfileError(f"{name} must be a list, not a string (no splitting)")
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileError(f"{name} must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ProfileError(f"{name}[{index}] must be a non-empty string")
        result.append(item)
    return result


def _fresh_absolute_dir(value: Any, name: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ProfileError(f"{name} must be absolute")
    if path.exists():
        raise ProfileError(f"{name} must be fresh (already exists)")
    return path


def build_collector_argv(
    app_argv: Sequence[str],
    output_dir: str | Path,
    *,
    msprof: str = MSPROF,
) -> list[str]:
    """Build ``msprof [fixed args] <app> [app args]`` as a list.

    The app argv is kept byte-for-byte; no shell, no split, no re-ordering.
    """
    app = _string_list(app_argv, "app_argv")
    if not isinstance(msprof, str) or not msprof:
        raise ProfileError("msprof must be a non-empty string")
    out = _fresh_absolute_dir(output_dir, "output_dir")
    prefix = [msprof, "--output", str(out)]
    for name, value in FIXED_ARGS:
        prefix.extend([name, value])
    forbidden = [name for name in FORBIDDEN_COLLECTOR_FLAGS if name in prefix]
    if forbidden:
        raise ProfileError(f"forbidden collector flag(s): {forbidden}")
    return prefix + app


def build_child_argv_json(
    app_argv: Sequence[str],
    output_dir: str | Path,
    *,
    msprof: str = MSPROF,
) -> str:
    return json.dumps(build_collector_argv(app_argv, output_dir, msprof=msprof))


def _as_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProfileError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    raise ProfileError(f"{name} must be an integer or null")


def _conversion_reasons(
    raw_ns: Optional[int],
    converted_ns: Optional[int],
    offset_ns: Optional[int],
    raw_clock: Optional[str],
    clock: Optional[str],
) -> set[str]:
    reasons: set[str] = set()
    if raw_ns is not None and converted_ns is not None and offset_ns is not None:
        if converted_ns != raw_ns + offset_ns:
            reasons.add("conversion_mismatch")
    if raw_clock is not None and clock is not None and raw_clock == clock:
        if offset_ns not in (None, 0):
            reasons.add("identity_offset_nonzero")
    return reasons


@dataclass(frozen=True)
class ExpectedTarget:
    """Fresh task identity and explicit raw/offset clock binding."""

    kernel_name: Optional[str]
    stream_id: Optional[int]
    task_id: Optional[int]
    raw_submit_ns: Optional[int]
    raw_end_ns: Optional[int]
    submit_ns: Optional[int]
    end_ns: Optional[int]
    end_kind: Optional[str]
    raw_clock: Optional[str]
    clock: Optional[str]
    offset_ns: Optional[int]
    clock_provenance: Optional[str]
    source: Optional[str]
    provenance: Optional[str]
    ambiguities: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpectedTarget":
        if not isinstance(value, Mapping):
            raise ProfileError("target must be an object")
        return cls(
            kernel_name=value.get("kernel_name"),
            stream_id=_as_int(value.get("stream_id"), "stream_id"),
            task_id=_as_int(value.get("task_id"), "task_id"),
            raw_submit_ns=_as_int(value.get("raw_submit_ns"), "raw_submit_ns"),
            raw_end_ns=_as_int(value.get("raw_end_ns"), "raw_end_ns"),
            submit_ns=_as_int(value.get("submit_ns"), "submit_ns"),
            end_ns=_as_int(value.get("end_ns"), "end_ns"),
            end_kind=value.get("end_kind"),
            raw_clock=value.get("raw_clock"),
            clock=value.get("clock"),
            offset_ns=_as_int(value.get("offset_ns"), "offset_ns"),
            clock_provenance=value.get("clock_provenance"),
            source=value.get("source"),
            provenance=value.get("provenance"),
            ambiguities=tuple(value.get("ambiguities") or ()),
        )


def validate_target(target: ExpectedTarget) -> list[str]:
    """Return fail-closed reasons; an empty list means fully bound."""
    reasons: set[str] = set()
    checks = (
        ("kernel_name", target.kernel_name),
        ("stream_id", target.stream_id),
        ("task_id", target.task_id),
        ("raw_submit_ns", target.raw_submit_ns),
        ("raw_end_ns", target.raw_end_ns),
        ("submit_ns", target.submit_ns),
        ("end_ns", target.end_ns),
        ("end_kind", target.end_kind),
        ("raw_clock", target.raw_clock),
        ("clock", target.clock),
        ("offset_ns", target.offset_ns),
        ("clock_provenance", target.clock_provenance),
        ("source", target.source),
        ("provenance", target.provenance),
    )
    for name, value in checks:
        if value is None or value == "":
            reasons.add(f"missing_{name}")
    if target.ambiguities:
        reasons.add("ambiguous_target")
    for clock in (target.raw_clock, target.clock):
        if clock is not None and clock not in KNOWN_CLOCKS:
            reasons.add("unknown_clock")
    if target.end_kind is not None and target.end_kind not in END_KINDS:
        reasons.add("unknown_end_kind")
    reasons.update(
        _conversion_reasons(
            target.raw_submit_ns,
            target.submit_ns,
            target.offset_ns,
            target.raw_clock,
            target.clock,
        )
    )
    reasons.update(
        _conversion_reasons(
            target.raw_end_ns,
            target.end_ns,
            target.offset_ns,
            target.raw_clock,
            target.clock,
        )
    )
    if (
        target.submit_ns is not None
        and target.end_ns is not None
        and not (target.submit_ns < target.end_ns)
    ):
        reasons.add("invalid_window")
    return sorted(reasons)


@dataclass(frozen=True)
class SampleRecord:
    """Explicit internal record with raw timestamp and offset conversion."""

    valid: bool
    kernel_name: Optional[str]
    raw_timestamp_ns: Optional[int]
    timestamp_ns: Optional[int]
    stream_id: Optional[int]
    task_id: Optional[int]
    raw_clock: Optional[str]
    clock: Optional[str]
    offset_ns: Optional[int]
    clock_provenance: Optional[str]
    source: str
    device: Optional[str] = None
    aic_sample: bool = True
    label: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleRecord":
        if not isinstance(value, Mapping):
            raise ProfileError("sample record must be an object")
        required = (
            "valid",
            "kernel_name",
            "raw_timestamp_ns",
            "timestamp_ns",
            "stream_id",
            "task_id",
            "raw_clock",
            "clock",
            "offset_ns",
            "clock_provenance",
            "source",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ProfileError(f"sample record missing key(s): {missing}")
        if not isinstance(value["valid"], bool):
            raise ProfileError("sample valid must be bool")
        return cls(
            valid=value["valid"],
            kernel_name=value.get("kernel_name"),
            raw_timestamp_ns=_as_int(value.get("raw_timestamp_ns"), "raw_timestamp_ns"),
            timestamp_ns=_as_int(value.get("timestamp_ns"), "timestamp_ns"),
            stream_id=_as_int(value.get("stream_id"), "stream_id"),
            task_id=_as_int(value.get("task_id"), "task_id"),
            raw_clock=value.get("raw_clock"),
            clock=value.get("clock"),
            offset_ns=_as_int(value.get("offset_ns"), "offset_ns"),
            clock_provenance=value.get("clock_provenance"),
            source=str(value["source"]),
            device=value.get("device"),
            aic_sample=bool(value.get("aic_sample", True)),
            label=value.get("label"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterResult:
    records: tuple[SampleRecord, ...] = ()
    reasons: tuple[str, ...] = ()
    mock_only: bool = False
    real: bool = False
    timeline_only: bool = False
    output_files: tuple[dict[str, Any], ...] = ()


def _artifact_list(directory: Path) -> tuple[dict[str, Any], ...]:
    if not directory.exists():
        return ()
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        data = path.read_bytes()
        rows.append(
            {
                "path": str(path),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return tuple(rows)


class SyntheticAdapter:
    """Read an explicitly labelled ``mock_only`` synthetic sample file."""

    @staticmethod
    def load(path: str | Path) -> AdapterResult:
        try:
            value = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return AdapterResult(reasons=(f"invalid_synthetic_file: {exc}",))
        if not isinstance(value, Mapping) or value.get("mock_only") is not True:
            return AdapterResult(reasons=("synthetic_not_labeled",))
        raw_records = value.get("records")
        if not isinstance(raw_records, list):
            return AdapterResult(reasons=("invalid_synthetic_record",), mock_only=True)
        records: list[SampleRecord] = []
        reasons: list[str] = []
        for index, raw in enumerate(raw_records):
            try:
                record = SampleRecord.from_dict(raw)
            except ProfileError as exc:
                reasons.append(f"invalid_synthetic_record[{index}]: {exc}")
                continue
            if record.source != "synthetic":
                reasons.append(f"invalid_synthetic_record[{index}]: source must be synthetic")
                continue
            records.append(record)
        files = value.get("output_files")
        output_files = tuple(files) if isinstance(files, list) else ()
        return AdapterResult(
            records=tuple(records),
            reasons=tuple(reasons) + (("timeline_only",) if value.get("timeline_only") else ()),
            mock_only=True,
            real=False,
            timeline_only=bool(value.get("timeline_only")),
            output_files=output_files,
        )


class RealOutputAdapter:
    """Placeholder for a future verified msprof-output schema.

    The current help has no field schema; this adapter deliberately does not
    parse or invent one.  It returns ``unknown_schema`` and the raw artifact
    list so a reviewer can inspect the original output.
    """

    @staticmethod
    def load(output_dir: str | Path) -> AdapterResult:
        directory = Path(output_dir)
        if not directory.exists():
            return AdapterResult(reasons=("output_missing", "unknown_schema"), real=True)
        files = _artifact_list(directory)
        if not files:
            return AdapterResult(reasons=("output_missing", "unknown_schema"), real=True)
        return AdapterResult(
            records=(),
            reasons=("unknown_schema",),
            mock_only=False,
            real=True,
            output_files=files,
        )


def _match_record(target: ExpectedTarget, record: SampleRecord) -> tuple[bool, set[str]]:
    reasons: set[str] = set()
    if not record.valid or not record.aic_sample:
        reasons.add("sample_invalid")
    for name, value in (
        ("sample_kernel", record.kernel_name),
        ("sample_raw_timestamp", record.raw_timestamp_ns),
        ("sample_timestamp", record.timestamp_ns),
        ("sample_stream", record.stream_id),
        ("sample_task", record.task_id),
        ("sample_raw_clock", record.raw_clock),
        ("sample_clock", record.clock),
        ("sample_offset", record.offset_ns),
        ("sample_clock_provenance", record.clock_provenance),
    ):
        if value is None or value == "":
            reasons.add(f"missing_{name}")
    if record.clock is not None and target.clock is not None and record.clock != target.clock:
        reasons.add("clock_mismatch")
    reasons.update(
        _conversion_reasons(
            record.raw_timestamp_ns,
            record.timestamp_ns,
            record.offset_ns,
            record.raw_clock,
            record.clock,
        )
    )
    if record.kernel_name is not None and target.kernel_name is not None:
        if record.kernel_name != target.kernel_name:
            reasons.add("kernel_mismatch")
    if record.stream_id is not None and target.stream_id is not None:
        if record.stream_id != target.stream_id:
            reasons.add("stream_mismatch")
    if record.task_id is not None and target.task_id is not None:
        if record.task_id != target.task_id:
            reasons.add("task_mismatch")
    if (
        record.timestamp_ns is not None
        and target.submit_ns is not None
        and target.end_ns is not None
    ):
        if record.timestamp_ns < target.submit_ns:
            reasons.add("before_submit")
        elif record.timestamp_ns > target.end_ns:
            reasons.add("after_end")
    return not reasons, reasons


def coverage_model() -> dict[str, Any]:
    return {
        "scope": "early_pending_only",
        "delay_s": DELAY_S,
        "duration_s": DURATION_S,
        "total_s": TOTAL_S,
        "timeout_approx_s": TIMEOUT_APPROX_S,
        "covers_timeout": False,
        "covers_late_pending": False,
        "unproven": "duration arrival flush/exit/kill/wait semantics are not documented",
    }


def classify_profile(
    target: ExpectedTarget,
    adapter: AdapterResult,
    collector: Mapping[str, Any],
    *,
    harness_checks: Optional[Mapping[str, bool]] = None,
    coverage: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Fail-closed classification of mock/synthetic outcomes.

    ``capability_proven`` is always false here.  CPU synthetic evidence can only
    make ``harness_pass`` true; device raw artifacts must be reviewed manually
    before any capability claim.
    """
    reasons: set[str] = set(adapter.reasons)
    if adapter.mock_only:
        reasons.add("mock_only")
    if adapter.real:
        reasons.add("real_adapter")
    reasons.update(validate_target(target))
    for key in ("timed_out", "truncated", "output_missing", "collector_exited_early", "ranks_alive"):
        if collector.get(key):
            reasons.add(key)
    if collector.get("timed_out"):
        reasons.add("collector_timeout")
    if collector.get("exit_code") not in (None, 0):
        reasons.add("collector_error")
    if adapter.timeline_only:
        reasons.add("timeline_only")
    if not adapter.records and not adapter.timeline_only:
        reasons.add("sample_empty")

    matches: list[SampleRecord] = []
    mismatch: set[str] = set()
    for record in adapter.records:
        matched, record_reasons = _match_record(target, record)
        if matched:
            matches.append(record)
        else:
            mismatch.update(record_reasons)

    record_unproven = mismatch & (HARD_COLLECTOR_REASONS | MISSING_REASONS)
    hard = (
        bool(reasons & HARD_COLLECTOR_REASONS)
        or bool(record_unproven)
        or any(
            reason.startswith("invalid_synthetic_record")
            or reason.startswith("invalid_synthetic_file")
            for reason in reasons
        )
    )
    missing = bool(reasons & MISSING_REASONS)
    if hard or missing or not matches:
        if not matches and not (hard or missing):
            status = STATUS_NOT_SUCCESS
            reasons.update(mismatch)
        else:
            status = STATUS_UNPROVEN
            if not matches:
                reasons.update(mismatch)
    elif len(matches) > 1:
        status = STATUS_UNPROVEN
        reasons.add("multiple_candidates")
    else:
        status = STATUS_CANDIDATE

    harness_pass = bool(harness_checks) and all(bool(v) for v in harness_checks.values())
    result = {
        "schema": "grouped-mmar-sample-profile-result/1",
        "status": status,
        "capability_proven": False,
        "capability_demonstrated": False,
        "manual_device_review_required": status == STATUS_CANDIDATE and not adapter.mock_only,
        "mock_candidate": status == STATUS_CANDIDATE and adapter.mock_only,
        "harness_pass": harness_pass,
        "mock_only": bool(adapter.mock_only),
        "real_adapter": bool(adapter.real),
        "reasons": sorted(reasons),
        "matches": len(matches),
        "target": asdict(target),
        "adapter": {
            "records": len(adapter.records),
            "reasons": sorted(set(adapter.reasons)),
            "mock_only": bool(adapter.mock_only),
            "real": bool(adapter.real),
            "timeline_only": bool(adapter.timeline_only),
            "output_files": list(adapter.output_files),
        },
        "collector": dict(collector),
        "coverage": dict(coverage or coverage_model()),
        "kernel_counter_evidence": False,
        "not_kernel_counter_evidence": True,
    }
    return result


def classify_lifecycle(
    *,
    collector_exit: str = "clean",
    app_exit: str = "clean",
    ranks_alive: bool = False,
    output_present: bool = False,
    raw_artifacts_retained: bool = True,
    restore_count: int = 0,
    gate_fail_closed: bool = False,
) -> dict[str, Any]:
    """Model the collector/app/torchrun/rank lifecycle outcomes.

    This is a CPU model only.  Actual tree kill/settle/gate/begin_restore remains
    the existing 8f4b843 supervisor implementation.
    """
    reasons: set[str] = set()
    if collector_exit == "early":
        reasons.add("collector_exited_early")
    elif collector_exit == "timeout":
        reasons.add("collector_timeout")
    elif collector_exit not in ("clean",):
        reasons.add("collector_error")
    if app_exit not in ("clean",):
        reasons.add("app_not_clean")
    if ranks_alive:
        reasons.add("ranks_alive")
    if not output_present:
        reasons.add("output_missing")
    if not raw_artifacts_retained:
        reasons.add("raw_artifacts_missing")
    if restore_count > 1:
        reasons.add("multiple_restores")
        gate_fail_closed = True
    cleanup_required = ranks_alive or collector_exit in ("early", "timeout")
    restore_required = cleanup_required or gate_fail_closed
    if restore_required and restore_count != 1:
        reasons.add("restore_missing" if restore_count == 0 else "restore_invalid")
    if gate_fail_closed:
        reasons.add("gate_fail_closed")
    return {
        "schema": "grouped-mmar-sample-profile-lifecycle/1",
        "cleanup_required": cleanup_required,
        "restore_required": restore_required,
        "restore_count": restore_count,
        "one_restore_ok": restore_required and restore_count == 1 and not gate_fail_closed,
        "gate_fail_closed": gate_fail_closed,
        "raw_artifacts_retained": raw_artifacts_retained,
        "reasons": sorted(reasons),
        "harness_model_only": True,
    }


def build_plan_skeleton(
    *,
    output_dir: str = "<fresh-absolute-dir>",
    msprof: str = MSPROF,
) -> dict[str, Any]:
    return {
        "schema": "grouped-mmar-sample-profile-plan/1",
        "status": "cpu_prepared_not_deployed",
        "mock_only": True,
        "device_go": False,
        "collector": {
            "msprof": msprof,
            "version": MSPROF_VERSION,
            "sha256": MSPROF_SHA256,
            "help_package": HELP_PACKAGE,
            "mode": "sample-based",
            "output_dir_creation": "helper validates fresh/absolute only; supervisor/msprof creates it; missing output = unproven",
            "command_prefix": [
                msprof,
                "--output",
                output_dir,
                *[item for pair in FIXED_ARGS for item in pair],
            ],
        },
        "flags": flag_specs(),
        "app": {
            "argv_list_only": True,
            "shell": False,
            "source": "supervised_probe_window.build_child_argv (unchanged)",
            "native_binding_sha256": "101660f812306177f6d09c77644492f8f1b4a7f08ec6f22fc54e3caf7986ce3f",
            "native_run_probe_sha256": "8e3088042ad771c4d9a1b51d94ad1f49ec918c078d0b80ee25e8b3ca99cddf2f",
            "adapter_sha256": "d181b15ab75e25982e527303de616c76a29e8f162038eb9f7dfe2a0c364110c3",
            "app_args_boundary": "explicit list; fixed app command fixture must verify no split/consume",
        },
        "target": {
            "fields": list(REQUIRED_TARGET_FIELDS),
            "values": {name: None for name in REQUIRED_TARGET_FIELDS},
            "source_required": "fresh StreamLaunchKernelV2 / ConstructFftsMixSqe / AllocTaskAndSendStars or fresh trace",
            "no_hardcode": True,
            "fail_closed_on_missing_or_ambiguous": True,
            "note": "old stream/task/kernel labels are historical; this run must bind its own",
        },
        "clock": {
            "submit_clock": "host plog UTC",
            "collector_clock": "collector timestamps",
            "sample_clock": "collector",
            "offset_required": True,
            "raw_and_converted": True,
            "same_clock_required_for_window": True,
            "clock_provenance_required": True,
            "identity_conversion": "raw_clock == clock requires offset_ns == 0 and a provenance",
            "different_domain_conversion": "converted == raw + offset; provenance non-empty",
            "same_label_without_offset": "capability_unproven",
            "unproven_if_offset_missing": True,
        },
        "coverage": coverage_model(),
        "success": {
            "and": [
                "valid AIC sample",
                "kernel identity",
                "sample time in this run submit-end window",
                "raw/offset/clock_provenance conversion validated",
                "this run stream/task attribution",
            ],
            "kernel_name_only": False,
            "timeline_only": False,
            "empty_truncated_error_timeout": STATUS_UNPROVEN,
            "capability_proven_default": False,
            "capability_demonstrated_requires_manual_device_review": True,
        },
        "classification": {
            "statuses": [STATUS_CANDIDATE, STATUS_UNPROVEN, STATUS_NOT_SUCCESS],
            "capability_demonstrated": "only after manual review of device raw artifacts",
            "mock_only": True,
            "harness_pass_separate": True,
            "no_auto_pass": True,
            "reasons": sorted(MISSING_REASONS | HARD_COLLECTOR_REASONS),
        },
        "lifecycle": {
            "tree": ["msprof", "app", "torchrun", "ranks"],
            "outer_timeout_s": OUTER_TIMEOUT_S,
            "collector_exit_flush_semantics": "unproven",
            "settle": "two consecutive empty rounds",
            "gate": "fail-closed",
            "restore": "exactly one begin_restore",
            "no_second_tree": True,
            "no_same_cell_retry": True,
            "raw_artifacts": "retain stdout/output even on failure",
            "supervisor": "supervised_probe_window.py 8f4b843 baseline; child-argv hook unchanged",
        },
        "freeze": {
            "baseline_commit": "8f4b843",
            "no_8ec": True,
            "no_minus2_go": True,
            "no_device": True,
            "no_workspace_read": True,
            "no_dynamic_attach": True,
            "no_instr_or_source": True,
        },
        "tests": {
            "suite": "test_sample_profile_cpu.py",
            "existing_50_not_rerun": True,
            "mock_only": True,
        },
    }


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def _cmd_child_argv(args: argparse.Namespace) -> int:
    raw = args.app_argv_json
    app_argv = json.loads(raw) if raw.lstrip().startswith("[") else _load_json(raw)
    print(json.dumps(build_collector_argv(app_argv, args.output, msprof=args.msprof)))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    print(json.dumps(build_plan_skeleton(output_dir=args.output, msprof=args.msprof), indent=2, ensure_ascii=False))
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    target = ExpectedTarget.from_dict(_load_json(args.target))
    collector = _load_json(args.collector)
    if args.real_output_dir:
        adapter = RealOutputAdapter.load(args.real_output_dir)
    else:
        adapter = SyntheticAdapter.load(args.samples)
    harness_checks = _load_json(args.harness_checks) if args.harness_checks else None
    result = classify_profile(target, adapter, collector, harness_checks=harness_checks)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("child-argv", help="print the msprof list argv; never executes it")
    child.add_argument("--app-argv-json", required=True)
    child.add_argument("--output", required=True)
    child.add_argument("--msprof", default=MSPROF)
    child.set_defaults(func=_cmd_child_argv)
    plan = sub.add_parser("plan-skeleton", help="print the plan skeleton")
    plan.add_argument("--output", default="<fresh-absolute-dir>")
    plan.add_argument("--msprof", default=MSPROF)
    plan.set_defaults(func=_cmd_plan)
    classify = sub.add_parser("classify", help="classify synthetic or real-adapter records")
    classify.add_argument("--target", required=True)
    classify.add_argument("--collector", required=True)
    group = classify.add_mutually_exclusive_group(required=True)
    group.add_argument("--samples")
    group.add_argument("--real-output-dir")
    classify.add_argument("--harness-checks")
    classify.set_defaults(func=_cmd_classify)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
