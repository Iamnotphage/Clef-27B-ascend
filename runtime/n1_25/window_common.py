"""Frozen N1 v2 window identity; CPU only, no library loading."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

VENDOR = "clef_native_ar_norm25"
IDENTITY_SHA = "1fde7e83d036a1032ed7455dc560303bcb9cfba5fa70769a7aff2359b64d89f6"
PREIMAGE_SHA = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
STOP_SHA = "d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886"
RUNNER_SHAS = {
    "run_cell.py": "e0352dcbfb7ff56f2a0e94d3e3052d3e1976049d7f9eaf9102d1880d8949b2f7",
    "contract.py": "8d1b39df08970829a502389112d4673ad920e68a620e2191173346b3de0b8f64",
    "fixed_gate.py": "041a66aa81c52d2a53bd825cf406323d7185759242669dff64b3551f86eba849",
}
CELL = {"rows": 967, "seed": 20261118, "cores": 2, "use_hccl": 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_sha(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA mismatch: {path}: {actual} != {expected}")


def frozen(stage):
    stage = Path(stage)
    require_sha(stage / "build_identity.json", IDENTITY_SHA)
    identity = json.loads((stage / "build_identity.json").read_text())
    require_sha(stage / "package.run", identity["package_sha256"])
    require_sha(stage / "n1_adapter.so", identity["adapter_sha256"])
    for name, sha in RUNNER_SHAS.items():
        require_sha(stage / name, sha)
    return identity


def archive_bytes(package, identity):
    """Read only: validate the exact package and every installable vendor byte.

    Materialize the one in-vendor relative symlink as identical file bytes.
    Never execute the package, extract arbitrary paths or alter toolkit state.
    """
    require_sha(package, identity["package_sha256"])
    raw = Path(package).read_bytes()
    match = re.search(rb'^skip="([0-9]+)"$', raw[:24000], re.M)
    if match is None:
        raise ValueError("missing Makeself header length")
    payload = raw.split(b"\n", int(match[1]))[-1]
    prefix = f"./packages/vendors/{VENDOR}/"
    result = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member")
        for rel, expected in identity["vendor_files"].items():
            if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
                raise ValueError("unsafe manifest path")
            member = archive.getmember(prefix + rel)
            if member.issym():
                if rel != "op_impl/ai_core/tbe/op_tiling/liboptiling.so" or member.linkname != "lib/linux/aarch64/libcust_opmaster_rt2.0.so":
                    raise ValueError("unexpected symlink")
            elif not member.isfile():
                raise ValueError("unexpected archive member type")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"archive vendor SHA mismatch: {rel}")
            result[rel] = data
        actual = {m.name[len(prefix):] for m in members if m.name.startswith(prefix) and not m.isdir()}
        if actual != set(result):
            raise ValueError("unmanifested vendor archive member")
    return result


def check_head(tree):
    require_sha(Path(tree) / "vllm/model_executor/models/clef_joint_schema.py", PREIMAGE_SHA)


def staged_paths(script, run_dir):
    stage = Path(script).resolve().parent
    run = Path(run_dir).resolve()
    if stage != run / "stage/runtime/n1_25":
        raise ValueError("N1 must execute from the supervised staged payload")
    return stage, run
