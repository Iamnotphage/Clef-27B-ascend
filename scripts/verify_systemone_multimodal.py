#!/usr/bin/env python3
"""Create deterministic RGB fixtures and check SystemOne image/video facts.

Plan locally without requests or model imports:
  python scripts/verify_systemone_multimodal.py --dry-run \
      --fixtures-dir /tmp/clef-mm-fixtures-20261112 --report /tmp/mm-plan.json
Before inference, --dry-run --decode-check tests the real installed vLLM
ImageMediaIO/VideoMediaIO without loading model weights. Use --modality image
for the initial image=1/video=0 service, then --modality all for the MM stage.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib

from verify_systemone import finite_float, get_health, reject_constant, validate


COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255)}
MM_PROCESSOR_KWARGS = {
    "images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 65536}},
    "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 65536}, "do_sample_frames": False},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def png(rgb, width=256, height=256):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    raw = (b"\0" + bytes(rgb) * width) * height
    # Stored DEFLATE blocks make bytes stable across compressor versions.
    stream = bytearray(b"\x78\x01")
    for offset in range(0, len(raw), 65535):
        block = raw[offset:offset + 65535]
        stream.extend(bytes([int(offset + len(block) == len(raw))]))
        stream.extend(struct.pack("<HH", len(block), len(block) ^ 0xFFFF))
        stream.extend(block)
    stream.extend(struct.pack(">I", zlib.adler32(raw) & 0xFFFFFFFF))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", bytes(stream)) + chunk(b"IEND", b"")


def avi(colors, width=96, height=96, fps=2):
    """RIFF AVI, uncompressed 24-bit DIB, keyframes, no audio or timestamps."""
    def chunk(kind, payload):
        return kind + struct.pack("<I", len(payload)) + payload + b"\0" * (len(payload) % 2)
    def listing(kind, payload):
        return chunk(b"LIST", kind + payload)
    stride = (width * 3 + 3) // 4 * 4
    frame_bytes = stride * height
    frames = [(bytes(reversed(COLORS[color])) * width + b"\0" * (stride - width * 3)) * height for color in colors]
    avih = struct.pack("<14I", 1_000_000 // fps, frame_bytes * fps, 0, 0x10, len(frames), 0, 1, frame_bytes, width, height, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHH8I4h", b"vids", b"DIB ", 0, 0, 0, 0, 1, fps, 0, len(frames), frame_bytes, 0xFFFFFFFF, 0, 0, 0, width, height)
    strf = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, frame_bytes, 0, 0, 0, 0)
    header = listing(b"hdrl", chunk(b"avih", avih) + listing(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movie, index, offset = bytearray(), bytearray(), 4
    for frame in frames:
        encoded = chunk(b"00db", frame)
        movie.extend(encoded)
        index.extend(struct.pack("<4sIII", b"00db", 0x10, offset, len(frame)))
        offset += len(encoded)
    payload = b"AVI " + header + listing(b"movi", bytes(movie)) + chunk(b"idx1", bytes(index))
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


def prepare_fixtures(directory):
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    definitions = [(f"{name}.png", png(rgb), "image/png", [name], 256) for name, rgb in COLORS.items()]
    definitions += [(f"{first}_{second}.avi", avi([first, second]), "video/x-msvideo", [first, second], 96) for first, second in (("red", "blue"), ("blue", "red"))]
    for name, data, media_type, colors, size in definitions:
        path = directory / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Existing fixture differs from deterministic bytes: {path}")
        path.write_bytes(data)
        files[name] = {
            "path": str(path.resolve()), "sha256": digest(data), "bytes": len(data),
            "media_type": media_type, "width": size, "height": size,
            "frame_colors_rgb": [list(COLORS[color]) for color in colors],
            "frames": len(colors), "fps": 2 if name.endswith(".avi") else None,
            "frame_pts_seconds": [0.0, 0.5] if name.endswith(".avi") else None,
            "duration_seconds": 1.0 if name.endswith(".avi") else None,
            "rgb_content_sha256": digest(b"".join(bytes(COLORS[color]) * size * size for color in colors)),
        }
    manifest = {
        "fixture_version": 1, "generator_script_sha256": digest(Path(__file__).read_bytes()),
        "encoding": "RGB8 PNG with stored DEFLATE; BI_RGB BGR24 AVI with idx1, two keyframes at 2fps, no audio",
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def planned_cases(model, manifest, modality):
    for filename, media in manifest["files"].items():
        is_image = media["media_type"].startswith("image/")
        if modality != "all" and (modality == "image") != is_image:
            continue
        field = "color" if is_image else "order"
        question = {
            "type": "choice",
            "instructions": "What is the solid color of the attached image?" if is_image else "Which color appears first and which appears second in the attached video?",
            "criteria": {"red": "Solid red", "blue": "Solid blue"} if is_image else {"red_then_blue": "Red first, then blue", "blue_then_red": "Blue first, then red"},
        }
        content = base64.b64encode(Path(media["path"]).read_bytes()).decode("ascii")
        request = {
            "model": model,
            "state": "Inspect the attached image." if is_image else "Inspect the attached video in chronological frame order.",
            "questions": {field: question},
            "images" if is_image else "videos": [f"data:{media['media_type']};base64,{content}"],
        }
        expected = filename.removesuffix(".png") if is_image else filename.removesuffix(".avi").replace("_", "_then_")
        yield {"name": filename, "question_id": field, "expected_choice": expected, "fixture_sha256": media["sha256"], "request": request}


def decode_check(cases, manifest):
    # This optional path calls the same media loaders as ClefFlashStrategy.
    from vllm.multimodal.media import ImageMediaIO, VideoMediaIO
    import cv2
    results = []
    for case in cases:
        media = manifest["files"][case["name"]]
        data = Path(media["path"]).read_bytes()
        if media["media_type"].startswith("image/"):
            image = ImageMediaIO().load_bytes(data).media
            extrema = [list(pair) for pair in image.getextrema()]
            expected = [[channel, channel] for channel in media["frame_colors_rgb"][0]]
            result = {"name": case["name"], "size": list(image.size), "mode": image.mode, "channel_extrema": extrema, "passed": image.size == (256, 256) and image.mode == "RGB" and extrema == expected}
        else:
            frames, metadata = VideoMediaIO(ImageMediaIO(), num_frames=-1, fps=2, video_backend="opencv").load_bytes(data).media
            extrema = [[[int(frame[..., channel].min()), int(frame[..., channel].max())] for channel in range(3)] for frame in frames]
            expected = [[[channel, channel] for channel in color] for color in media["frame_colors_rgb"]]
            result = {"name": case["name"], "shape": list(frames.shape), "metadata": metadata, "channel_extrema": extrema, "passed": tuple(frames.shape) == (2, 96, 96, 3) and extrema == expected and metadata["fps"] == 2}
        results.append(result)
    return {"passed": all(item["passed"] for item in results), "opencv_version": cv2.__version__, "stream_buffered_backends": [int(item) for item in cv2.videoio_registry.getStreamBufferedBackends()], "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("/tmp/clef-mm-fixtures-20261112"))
    parser.add_argument("--modality", "--only", choices=("image", "video", "all"), default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="clef")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decode-check", action="store_true", help="Check installed vLLM decoders without loading weights")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.atol) or args.atol < 0:
        parser.error("Timeout must be positive; atol must be nonnegative; both must be finite")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "passed": False,
        "status": "preparing", "dry_run": args.dry_run, "modality": args.modality,
        "base_url": args.base_url.rstrip("/"), "cases": [],
        "recommended_server_configuration": {
            "limit_mm_per_prompt": {"image": 1, "video": 1}, "skip_mm_profiling": True,
            "mm_processor_kwargs": MM_PROCESSOR_KWARGS,
            "VLLM_VIDEO_LOADER_BACKEND": "opencv", "max_model_len": 1024,
        },
        "source_token_budget": {
            "basis": "Fixed vLLM Qwen3VLProcessingInfo._get_vision_info; patch_size16, merge_size2, temporal_patch_size2; processor bounds shown above",
            "image_grid_thw": [1, 16, 16], "image_vision_tokens": 64,
            "video_grid_thw": [1, 6, 6], "video_vision_tokens": 9,
            "formula": "grid_t * grid_h * grid_w / merge_size**2",
            "exact_total_input_tokens": "Recorded from each actual API response; includes fixed prompt, state, schema, timestamps and expanded media placeholders",
            "model_config_sha256": "c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58",
            "processor_config_sha256": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        },
    }
    try:
        manifest = prepare_fixtures(args.fixtures_dir)
        report["fixture_manifest"] = manifest
        report["cases"] = list(planned_cases(args.model, manifest, args.modality))
        if args.decode_check:
            report["decoder_check"] = decode_check(report["cases"], manifest)
            if not report["decoder_check"]["passed"]:
                raise ValueError("Decoded frames differ from fixture dimensions, timing or exact colors")
        if args.dry_run:
            report["status"] = "planned"
        else:
            report["health"] = get_health(report["base_url"], 0, args.timeout)
            if not report["health"]["passed"]:
                raise ValueError("Service health check failed")
            for case in report["cases"]:
                case.update(passed=False, protocol_passed=False, semantic_passed=False)
                started = time.perf_counter()
                try:
                    body = json.dumps(case["request"], ensure_ascii=False).encode()
                    message = Request(report["base_url"] + "/v1/systemone", data=body, headers={"Content-Type": "application/json"}, method="POST")
                    with urlopen(message, timeout=args.timeout) as response:
                        case.update(http_status=response.status, response_text=response.read().decode())
                    case["response"] = json.loads(case["response_text"], parse_float=finite_float, parse_constant=reject_constant)
                    if not isinstance(case["response"], dict):
                        raise ValueError("Response JSON must be an object")
                    case["validation_errors"] = validate(case["request"], case["response"], args.atol)
                    case["protocol_passed"] = case["http_status"] == 200 and not case["validation_errors"]
                    if case["protocol_passed"]:
                        case["actual_choice"] = case["response"]["answers"][case["question_id"]]["choice"]
                        case["actual_input_tokens"] = case["response"]["usage"]["input_tokens"]
                        case["semantic_passed"] = case["actual_choice"] == case["expected_choice"]
                    case["passed"] = case["protocol_passed"] and case["semantic_passed"]
                except HTTPError as exc:
                    case.update(http_status=exc.code, response_text=exc.read().decode(errors="replace"), error=str(exc))
                except (OSError, URLError, ValueError, KeyError, TypeError) as exc:
                    case["error"] = f"{type(exc).__name__}: {exc}"
                case["elapsed_seconds"] = time.perf_counter() - started
            report["passed"] = all(case["passed"] for case in report["cases"])
            report["status"] = "complete" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "passed": report["passed"], "report": str(args.report.resolve()), "cases": [{key: case.get(key) for key in ("name", "expected_choice", "actual_choice", "passed", "actual_input_tokens")} for case in report["cases"]]}), flush=True)
    return 0 if report["status"] == "planned" or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
