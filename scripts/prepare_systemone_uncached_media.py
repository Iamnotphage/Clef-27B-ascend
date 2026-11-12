#!/usr/bin/env python3
"""Create fixed-shape, unique decoded-pixel media for cold vision profiling.

Warmup tags 1..10 and profiling tags 31..35 change one green channel byte of one
red pixel. The RGB PNG uses stored DEFLATE. The BI_RGB AVI changes one payload
byte only; all headers, chunk sizes and idx1 entries remain unchanged. No HTTP,
model, processor or NPU is used. Actual encoder-cache misses/vision execution
must subsequently be verified from the service trace and cache evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import zlib

from verify_systemone_multimodal import COLORS, MM_PROCESSOR_KWARGS, avi, planned_cases, png


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def stored_deflate(raw):
    stream = bytearray(b"\x78\x01")
    for offset in range(0, len(raw), 65535):
        block = raw[offset:offset + 65535]
        stream.extend(bytes([int(offset + len(block) == len(raw))]))
        stream.extend(struct.pack("<HH", len(block), len(block) ^ 0xFFFF))
        stream.extend(block)
    stream.extend(struct.pack(">I", zlib.adler32(raw) & 0xFFFFFFFF))
    return bytes(stream)


def decode_png(data):
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    offset, chunks = len(PNG_SIGNATURE), []
    while offset < len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        crc = struct.unpack_from(">I", data, offset + 8 + size)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise ValueError("PNG chunk CRC mismatch")
        chunks.append((kind, payload))
        offset += size + 12
    header = next(payload for kind, payload in chunks if kind == b"IHDR")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header)
    if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("Expected RGB8 noninterlaced PNG")
    raw = zlib.decompress(b"".join(payload for kind, payload in chunks if kind == b"IDAT"))
    stride = 1 + width * 3
    if len(raw) != stride * height or any(raw[row * stride] != 0 for row in range(height)):
        raise ValueError("Expected filter-0 PNG scanlines")
    pixels = b"".join(raw[row * stride + 1:(row + 1) * stride] for row in range(height))
    return width, height, raw, pixels, chunks


def tagged_png(tag):
    base = png(COLORS["red"], width=256, height=256)
    width, height, raw, base_pixels, chunks = decode_png(base)
    changed = bytearray(raw)
    changed[2] = tag  # Filter byte, R, then G of decoded pixel (y=0,x=0).
    data = PNG_SIGNATURE + b"".join(chunk(kind, stored_deflate(bytes(changed)) if kind == b"IDAT" else payload) for kind, payload in chunks)
    _, _, _, pixels, _ = decode_png(data)
    differences = [index for index, pair in enumerate(zip(base_pixels, pixels)) if pair[0] != pair[1]]
    if differences != [1] or pixels[:3] != bytes((255, tag, 0)):
        raise ValueError("PNG decoded-pixel tag verification failed")
    return data, {"width": width, "height": height, "frames": 1, "decoded_rgb_sha256": digest(pixels), "decoded_tag": {"frame_index": 0, "coordinate_yx": [0, 0], "rgb": list(pixels[:3]), "green_channel_tag": tag}, "decoded_channel_differences_vs_base": len(differences), "base_fixture_sha256": digest(base), "encoding": "RGB8 filter-0 PNG with stored DEFLATE; no metadata-only tag", "decoder": "stdlib PNG chunk CRC validation, zlib inflate and filter-0 RGB scanline decoding"}


def avi_payloads(data):
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise ValueError("Expected RIFF AVI")
    result = []
    offset = 12
    while offset + 8 <= len(data):
        kind = data[offset:offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start, end = offset + 8, offset + 8 + size
        if kind == b"LIST" and data[start:start + 4] == b"movi":
            cursor = start + 4
            while cursor + 8 <= end:
                frame_kind = data[cursor:cursor + 4]
                frame_size = struct.unpack_from("<I", data, cursor + 4)[0]
                if frame_kind == b"00db":
                    result.append((cursor + 8, frame_size))
                cursor += 8 + frame_size + frame_size % 2
        offset = end + size % 2
    if len(result) != 2:
        raise ValueError("Expected exactly two uncompressed AVI frames")
    return result


def decode_avi_pixels(data, width=96, height=96):
    stride = (width * 3 + 3) // 4 * 4
    frames = []
    for offset, size in avi_payloads(data):
        if size != stride * height:
            raise ValueError("AVI frame payload has unexpected shape")
        frame = bytearray()
        for row in reversed(range(height)):
            scanline = data[offset + row * stride:offset + row * stride + width * 3]
            for pixel in range(width):
                blue, green, red = scanline[pixel * 3:pixel * 3 + 3]
                frame.extend((red, green, blue))
        frames.append(bytes(frame))
    return frames


def tagged_avi(tag):
    base = avi(["red", "blue"], width=96, height=96, fps=2)
    first_payload, _ = avi_payloads(base)[0]
    offset = first_payload + (96 - 1) * (96 * 3) + 1
    changed = bytearray(base)
    changed[offset] = tag
    data = bytes(changed)
    byte_differences = [index for index, pair in enumerate(zip(base, data)) if pair[0] != pair[1]]
    frames = decode_avi_pixels(data)
    base_frames = decode_avi_pixels(base)
    if byte_differences != [offset] or frames[0][:3] != bytes((255, tag, 0)) or frames[1] != base_frames[1]:
        raise ValueError("AVI payload tag/header/index verification failed")
    return data, {"width": 96, "height": 96, "frames": 2, "fps": 2, "duration_seconds": 1, "frame_pts_seconds": [0, .5], "decoded_rgb_sha256": digest(b"".join(frames)), "decoded_frame_rgb_sha256": [digest(frame) for frame in frames], "decoded_tag": {"frame_index": 0, "coordinate_yx": [0, 0], "rgb": list(frames[0][:3]), "green_channel_tag": tag}, "decoded_channel_differences_vs_base": 1, "base_fixture_sha256": digest(base), "encoded_byte_differences_vs_base": 1, "modified_payload_byte_offset": offset, "header_and_idx_unchanged": True, "encoding": "Two-frame BI_RGB BGR24 AVI; one first-frame pixel G byte changed; headers, sizes and idx1 unchanged", "decoder": "stdlib RIFF/movi/00db parsing and bottom-up BI_RGB24 to top-down RGB8 decoding"}


def prepare(output, model="clef"):
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "generator_script_sha256": digest(Path(__file__).read_bytes()), "model": model, "passed": False, "http_requests_sent": False, "npu_operations_executed": False, "processor_kwargs": MM_PROCESSOR_KWARGS, "video_loader_backend": "opencv", "modalities": {}}
    for modality, generate, filename, media_type, expected_tokens in (("image", tagged_png, "red.png", "image/png", 203), ("video", tagged_avi, "red_blue.avi", "video/x-msvideo", 173)):
        directory = output / modality
        directory.mkdir(parents=True, exist_ok=True)
        phase_hashes, fixtures, phase_files = {}, [], {}
        for phase, tags in (("warmup", range(1, 11)), ("profile", range(31, 36))):
            samples, hashes = [], []
            for tag in tags:
                data, decoded = generate(tag)
                path = directory / (f"{phase}-tag-{tag:02d}-" + filename)
                if path.exists() and path.read_bytes() != data:
                    raise ValueError(f"Existing tagged fixture differs: {path}")
                path.write_bytes(data)
                fixture = {"path": str(path.resolve()), "bytes": len(data), "sha256": digest(data), "media_type": media_type, "phase": phase, "tag": tag, **decoded}
                fixtures.append(fixture)
                hashes.append(decoded["decoded_rgb_sha256"])
                case = next(planned_cases(model, {"files": {filename: fixture}}, modality))
                metadata = {"phase": phase, "tag": tag, "fixture": fixture, "expected_choice": case["expected_choice"], "expected_input_tokens": expected_tokens, "token_expectation_basis": "Previously validated identical shape/processor; actual API usage must be checked", "processor_kwargs": MM_PROCESSOR_KWARGS, "video_loader_backend": "opencv", "single_use_request": True, "cache_miss_design": "Unique decoded pixel RGB bytes, distinct from every warmup/profile request of this modality; actual encoder cache miss and vision kernel coverage require runtime evidence"}
                samples.append({"name": f"uncached-{modality}-{phase}-tag-{tag:02d}", "request": case["request"], "metadata": metadata})
            phase_path = directory / (phase + "-samples.json")
            phase_path.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2) + "\n")
            phase_files[phase] = {"path": str(phase_path.resolve()), "sha256": digest(phase_path.read_bytes()), "sample_count": len(samples), "request_policy": "Send every sample exactly once with benchmark warmup=0; do not cycle beyond sample_count"}
            phase_hashes[phase] = hashes
        unique = len(set(phase_hashes["warmup"] + phase_hashes["profile"])) == 15
        disjoint = not (set(phase_hashes["warmup"]) & set(phase_hashes["profile"]))
        if not unique or not disjoint:
            raise ValueError("Decoded content tags must be unique and phase-disjoint")
        report["modalities"][modality] = {"expected_input_tokens": expected_tokens, "expected_vision_tokens": 64 if modality == "image" else 9, "sample_files": phase_files, "fixtures": fixtures, "decoded_pixel_hashes_unique": unique, "warmup_profile_decoded_hashes_disjoint": disjoint}
    report["cache_miss_design_boundary"] = "Fixed resolution/frame count/processor warms shapes while all 15 decoded contents differ. No cache reset is performed. Preprocessing/encoder cache policy and realized vision execution must be verified from actual mm_hashes/cache metrics and NPU trace; tag design alone does not verify a miss"
    report["passed"] = True
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="clef")
    args = parser.parse_args()
    try:
        report = prepare(args.output_dir, args.model)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({"passed": report["passed"], "manifest": str((args.output_dir / "manifest.json").resolve()), "script_sha256": report["generator_script_sha256"], "modalities": {kind: {"sample_files": value["sample_files"], "unique_decoded_hashes": value["decoded_pixel_hashes_unique"], "phase_disjoint": value["warmup_profile_decoded_hashes_disjoint"]} for kind, value in report["modalities"].items()}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
