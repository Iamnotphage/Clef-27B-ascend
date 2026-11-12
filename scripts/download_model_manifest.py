#!/usr/bin/env python3
"""Download the fixed ModelScope manifest with resumable ranges and SHA-256 checks."""

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(entry, destination, chunk_bytes):
    relative = pathlib.PurePosixPath(entry["Path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid manifest path: {relative}")
    expected_size = entry["Size"]
    expected_hash = entry["Sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"Missing SHA-256 for {relative}")
    target = destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != expected_size or sha256_file(target) != expected_hash:
            raise ValueError(f"Existing file fails verification: {target}")
        print(json.dumps({"verified_existing": str(relative)}), flush=True)
        return {"path": str(relative), "bytes": expected_size, "sha256": expected_hash, "verified": True}

    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Partial file exceeds manifest size: {partial}")
    url = (
        "https://modelscope.cn/api/v1/models/Cloudflare/clef/repo?"
        + urllib.parse.urlencode({"Revision": entry["Revision"], "FilePath": str(relative)})
    )
    last_report = offset
    with partial.open("ab") as output:
        while offset < expected_size:
            end = min(offset + chunk_bytes, expected_size) - 1
            # Include the range in the URL to avoid stale intermediary range caches.
            request = urllib.request.Request(
                url + f"&clef_range={offset}-{end}",
                headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
            )
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        content_range = response.headers.get("Content-Range")
                        expected_range = f"bytes {offset}-{end}/{expected_size}"
                        full_response = offset == 0 and end + 1 == expected_size and response.status == 200
                        if content_range != expected_range and not (full_response and content_range is None):
                            raise ValueError(f"Unexpected range for {relative}: {content_range}")
                        data = response.read(end - offset + 2)
                        if len(data) != end - offset + 1:
                            raise ValueError(f"Incomplete range for {relative}: received {len(data)} bytes")
                    break
                except Exception as error:
                    if attempt == 5:
                        raise
                    print(json.dumps({"retry": str(relative), "offset": offset, "attempt": attempt + 1, "error": str(error)}), flush=True)
                    time.sleep(min(2 ** attempt, 20))
            output.write(data)
            output.flush()
            offset += len(data)
            if offset - last_report >= 256 * 1024 * 1024 or offset == expected_size:
                print(json.dumps({"progress": str(relative), "bytes": offset, "total": expected_size}), flush=True)
                last_report = offset
    actual_hash = sha256_file(partial)
    if actual_hash != expected_hash:
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch for {relative}: {actual_hash}")
    partial.replace(target)
    print(json.dumps({"verified": str(relative), "bytes": expected_size, "sha256": actual_hash}), flush=True)
    return {"path": str(relative), "bytes": expected_size, "sha256": actual_hash, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_mib < 1:
        parser.error("workers and chunk-mib must be positive")
    args.destination = args.destination.resolve()
    entries = [entry for entry in json.loads(args.manifest.read_text())["Data"]["Files"] if entry["Type"] == "blob"]
    if len({entry["Path"] for entry in entries}) != len(entries):
        parser.error("manifest contains duplicate paths")
    args.destination.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.destination.parent / f".{args.destination.name}.download.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_file, entry, args.destination, args.chunk_mib * 1024 * 1024) for entry in entries]
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    print(json.dumps({"fatal_error": str(error)}), flush=True)
                    raise
        report = {
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": "Cloudflare/clef",
            "destination": str(args.destination),
            "files": sorted(results, key=lambda item: item["path"]),
            "file_count": len(results),
            "total_bytes": sum(item["bytes"] for item in results),
            "all_sha256_verified": True,
            "manifest_sha256": sha256_file(args.manifest),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        temporary_report = args.report.with_suffix(".json.tmp")
        temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary_report.replace(args.report)
        print(json.dumps({"complete": True, "files": len(results), "bytes": report["total_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
