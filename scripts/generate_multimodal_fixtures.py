#!/usr/bin/env python3
"""Recreate the four byte-identical Clef image/video fixtures with stdlib.

Usage:
  python scripts/generate_multimodal_fixtures.py --output-dir /path/to/fixtures

The PNG/AVI algorithms are retained from the original verifier source recorded
in the 2026-11-12 fixture manifest. Each encoded file must match its original
size and SHA-256 before any output is written. The new manifest records this
standalone script's current SHA-256 and retains the original source, manifest
and archive hashes as provenance; its absolute paths follow --output-dir.
Reproduction needs only this script and the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib


COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255)}
DEFAULT_OUTPUT_DIR = Path("/workspace/yangchen/work/clef-mm-fixtures")
ENCODING = "RGB8 PNG with stored DEFLATE; BI_RGB BGR24 AVI with idx1, two keyframes at 2fps, no audio"
ORIGINAL_FILES = {
    "red.png": (196947, "f9500439461ef325070938a4b914334288855e68245212e2345eb1a125bcf27f"),
    "blue.png": (196947, "1d5fd2aadef8c5cdb384ba958dbf92ee33db178aeba929d00cc32dddf5712a58"),
    "red_blue.avi": (55576, "d504e03766ad216a29c33af846587bb9943bb72e2a35598c5458b3b4a6929862"),
    "blue_red.avi": (55576, "df7f6772fbe1a7ca660f620ba97bb0167c875d45e9a26d3fcc12ec42d98c9df7"),
}
ORIGINAL_PROVENANCE = {
    "generator_script": "scripts/verify_systemone_multimodal.py",
    "generator_script_sha256": "59b67194d42212bae20a1e0537c492f4f5d9cf9e9529c1bb2c9a69604622a2b3",
    "manifest": "clef-mm-fixtures-20261112/manifest.json",
    "manifest_sha256": "1a8da00182a9f20b174565c59da65a57afd1bc2dfe17f471df3d9b508bd076e3",
    "archive": "clef-mm-fixtures-20261112.tar.gz",
    "archive_bytes": 12775,
    "archive_sha256": "062471897379d23ed913bce77da3998e9b94bfa32917212ef3b52f34cf75f3a7",
    "migration_note": (
        "The top-level generator_script_sha256 identifies this standalone generator; "
        "this provenance identifies the original verifier source and fixture bundle. "
        "Output paths and the generator hash change after migration; all four encoded "
        "fixture sizes, hashes and media metadata remain identical. "
        "The original source, archive and temporary directory are optional evidence "
        "and are not required to regenerate the fixtures."
    ),
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
    directory = directory.expanduser().resolve()
    definitions = [(f"{name}.png", png(rgb), "image/png", [name], 256) for name, rgb in COLORS.items()]
    definitions += [(f"{first}_{second}.avi", avi([first, second]), "video/x-msvideo", [first, second], 96) for first, second in (("red", "blue"), ("blue", "red"))]
    # Validate the full set before creating files or replacing a manifest.
    for name, data, *_ in definitions:
        actual = len(data), digest(data)
        if actual != ORIGINAL_FILES[name]:
            raise ValueError(f"Generated fixture differs from the validated baseline: {name}: {actual}")
        path = directory / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Existing fixture differs from deterministic bytes: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data, media_type, colors, size in definitions:
        path = directory / name
        path.write_bytes(data)
        files[name] = {
            "path": str(path), "sha256": digest(data), "bytes": len(data),
            "media_type": media_type, "width": size, "height": size,
            "frame_colors_rgb": [list(COLORS[color]) for color in colors],
            "frames": len(colors), "fps": 2 if name.endswith(".avi") else None,
            "frame_pts_seconds": [0.0, 0.5] if name.endswith(".avi") else None,
            "duration_seconds": 1.0 if name.endswith(".avi") else None,
            "rgb_content_sha256": digest(b"".join(bytes(COLORS[color]) * size * size for color in colors)),
        }
    manifest = {
        "fixture_version": 1,
        "generator_script_sha256": digest(Path(__file__).read_bytes()),
        "encoding": ENCODING,
        "provenance": ORIGINAL_PROVENANCE,
        "all_original_fixture_hashes_verified": True,
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_fixtures(args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": "complete",
        "manifest": str(args.output_dir.expanduser().resolve() / "manifest.json"),
        "generator_script_sha256": manifest["generator_script_sha256"],
        "all_original_fixture_hashes_verified": True,
        "files": {name: {"bytes": item["bytes"], "sha256": item["sha256"]} for name, item in manifest["files"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
