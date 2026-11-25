"""Unit tests for runtime/apply_tiered_cache_patch.py.

The real preimages are the pinned vLLM sources; these tests pin synthetic
fixtures by patching the module's preimage constants, then verify that every
generated postimage strips back to the exact preimage bytes.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import apply_tiered_cache_patch as installer  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


MODEL_RUNNER_FIXTURE = '''import torch


class NPUModelRunner:
    def execute_model(self, scheduler_output):
        return scheduler_output
'''


API_ROUTER_FIXTURE = '''def build_routes():
    return {
        "cache": {
            "prefix_caching": False,
            "chunked_prefill": False,
        },
        "protocol": "2026-11-13",
    }
'''


def _pin(monkeypatch):
    monkeypatch.setattr(installer, "MODEL_RUNNER_PREIMAGE",
                        _sha(MODEL_RUNNER_FIXTURE))
    monkeypatch.setattr(installer, "API_ROUTER_PREIMAGE",
                        _sha(API_ROUTER_FIXTURE))


def test_model_runner_round_trip(monkeypatch):
    _pin(monkeypatch)
    postimage = installer.patch_model_runner(MODEL_RUNNER_FIXTURE)
    assert postimage.startswith(MODEL_RUNNER_FIXTURE)
    assert "NPUModelRunner.execute_model = _clef_tier2_execute_wrapper" in postimage
    assert installer.strip_model_runner(postimage) == MODEL_RUNNER_FIXTURE
    assert installer.strip_model_runner(MODEL_RUNNER_FIXTURE) is None


def test_api_router_inserts_disclosure_with_anchor_indent(monkeypatch):
    _pin(monkeypatch)
    postimage = installer.patch_api_router(API_ROUTER_FIXTURE)
    lines = postimage.splitlines()
    anchor = lines.index('            "chunked_prefill": False,')
    inserted = lines[anchor + 1]
    assert inserted.startswith('            "prefix_state_cache":')
    assert 'CLEF_PREFIX_STATE_CACHE' in inserted
    assert installer.strip_api_router(postimage) == API_ROUTER_FIXTURE
    assert installer.strip_api_router(API_ROUTER_FIXTURE) is None


def test_patch_refuses_unknown_source(monkeypatch):
    _pin(monkeypatch)
    try:
        installer.patch_model_runner(MODEL_RUNNER_FIXTURE + "# drift\n")
    except ValueError as exc:
        assert "verified preimage" in str(exc)
    else:
        raise AssertionError("unknown model_runner content was accepted")
    try:
        installer.patch_api_router(API_ROUTER_FIXTURE.replace(
            '"chunked_prefill": False,', '"chunked_prefill": True,'))
    except ValueError as exc:
        assert "verified preimage" in str(exc)
    else:
        raise AssertionError("unknown api_router content was accepted")


def test_api_router_requires_unique_anchor(monkeypatch):
    _pin(monkeypatch)
    duplicated = API_ROUTER_FIXTURE.replace(
        '"protocol": "2026-11-13",',
        '"chunked_prefill": False,')
    monkeypatch.setattr(installer, "API_ROUTER_PREIMAGE", _sha(duplicated))
    try:
        installer.patch_api_router(duplicated)
    except ValueError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("duplicate cache anchors were accepted")


def test_status_reports_tampered_content(monkeypatch, tmp_path):
    _pin(monkeypatch)
    target = tmp_path / "model_runner.py"
    target.write_text(MODEL_RUNNER_FIXTURE)
    assert installer.status(
        target, installer.MODEL_RUNNER_PREIMAGE,
        installer.strip_model_runner)[0] == "preimage"
    postimage = installer.patch_model_runner(MODEL_RUNNER_FIXTURE)
    target.write_text(postimage + "# drift\n")
    assert installer.status(
        target, installer.MODEL_RUNNER_PREIMAGE,
        installer.strip_model_runner)[0] == "unknown"
