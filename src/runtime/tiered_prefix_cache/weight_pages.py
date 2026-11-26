"""Drop cold safetensors page cache after the service has loaded weights.

The 27B checkpoint is ~54 GiB on disk.  Once vLLM has copied every shard into
NPU HBM, the kernel's clean file pages are pure reclaimable ballast: they
count against the cgroup and compete with the Tier 2A pageable representation
pool.  ``POSIX_FADV_DONTNEED`` on each read-only shard tells the kernel those
pages may be dropped; the call never changes the file and is safe to repeat.

The helper is deliberately free of torch/vLLM imports so the offline unit
tests can exercise the exact ``os.posix_fadvise(fd, 0, 0, DONTNEED)`` contract
on any POSIX host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class PageCacheReport:
    files: int = 0
    bytes: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "bytes": self.bytes,
            "skipped": list(self.skipped),
            "errors": list(self.errors),
        }


def drop_safetensors_page_cache(model_dir: str | os.PathLike | None, *,
                                apply=None) -> PageCacheReport:
    """Best-effort ``DONTNEED`` for every top-level ``*.safetensors`` file.

    Missing directories, absent ``posix_fadvise`` (non-POSIX host), and files
    that disappear concurrently are reported, never raised: weight loading has
    already succeeded, so cache hygiene must not take the service down.
    """
    report = PageCacheReport()
    if not model_dir:
        report.skipped.append("model_dir is unset")
        return report
    if apply is None:
        apply = hasattr(os, "posix_fadvise") and hasattr(
            os, "POSIX_FADV_DONTNEED")
    if not apply:
        report.skipped.append("posix_fadvise unavailable")
        return report
    directory = Path(model_dir).expanduser()
    try:
        paths = sorted(directory.glob("*.safetensors"))
    except OSError as exc:
        report.errors.append(f"{directory}: {type(exc).__name__}: {exc}")
        return report
    for path in paths:
        try:
            size = path.stat().st_size
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as exc:
            report.errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        report.files += 1
        report.bytes += size
    return report
