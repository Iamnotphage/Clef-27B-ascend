#!/usr/bin/env python3
"""Derive the audited aa3e candidate acceptance manifest from the frozen f810 one.

The candidate service differs from production f810 only in the source/guard/env
bindings that the deployment actually installs.  This generator enforces that
whitelist: any other leaf change in the manifest blocks.  It performs no NPU,
service or network action; the output is a static JSON document consumed by the
same frozen acceptance runner and supervisor.

Frozen inputs:
  base   final-acceptance cc7f3284 runtime/final_acceptance_manifest.json
         SHA-256 0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2
Preimage facts (read-only audited):
  f810 head  f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863
  aa3e head  aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503
  guard pre  76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9
  guard post 36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

BASE_SHA = '0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2'
F810_HEAD = 'f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863'
AA3E_HEAD = 'aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503'
GUARD_PRE = '76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9'
GUARD_POST = '36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7'
FLAG_ENV = 'CLEF_JOINT_FUSED_ATTN'
ALLOW_ENV = 'CLEF_JOINT_ALLOWED_POSTIMAGE_SHA'
HEAD_SUFFIX = '/vllm/model_executor/models/clef_joint_schema.py'
GUARD_SUFFIX = '/runtime/apply_joint_single_field_patch.py'
PHASES = ('s1', 's2_b8', 's2_b16', 'erqa')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class _Missing:
    def __repr__(self):
        return '<absent>'


MISSING = _Missing()


def leaf_diff(left, right, path=''):
    """Return {path: (old, new)}; dicts recurse, equal-length dict lists recurse."""
    if isinstance(left, dict) and isinstance(right, dict):
        changed = {}
        for key in sorted(set(left) | set(right)):
            changed.update(leaf_diff(left.get(key, MISSING), right.get(key, MISSING),
                                     f'{path}.{key}' if path else key))
        return changed
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        changed = {}
        for i, (a, b) in enumerate(zip(left, right)):
            changed.update(leaf_diff(a, b, f'{path}[{i}]'))
        return changed
    if isinstance(left, _Missing) or isinstance(right, _Missing):
        return {} if left is right else {path: (left, right)}
    return {} if left == right else {path: (left, right)}


def derive(base: dict) -> tuple[dict, list[str], dict]:
    m = copy.deepcopy(base)
    heads = [i for i, item in enumerate(m['source_files']) if item['path'].endswith(HEAD_SUFFIX)]
    guards = [i for i, item in enumerate(m['source_files']) if item['path'].endswith(GUARD_SUFFIX)]
    require(len(heads) == 1 and len(guards) == 1, 'unique head and guard source entries required')
    require(m['source_files'][heads[0]]['sha256'] == F810_HEAD, 'base head is not f810')
    require(m['source_files'][guards[0]]['sha256'] == GUARD_PRE, 'base guard is not the preimage')
    m['source_files'][heads[0]]['sha256'] = AA3E_HEAD
    m['source_files'][guards[0]]['sha256'] = GUARD_POST
    require(m['source_selection']['production_head_sha'] == F810_HEAD, 'source_selection drift')
    require(m['source_selection']['guard_sha'] == GUARD_PRE, 'source_selection guard drift')
    m['source_selection']['production_head_sha'] = AA3E_HEAD
    m['source_selection']['guard_sha'] = GUARD_POST
    require(FLAG_ENV in m['forbidden_environment'], 'base forbidden environment lacks the fused flag')
    m['forbidden_environment'] = [k for k in m['forbidden_environment'] if k != FLAG_ENV]
    for phase in PHASES:
        env = m['phases'][phase]['service_environment']
        require(FLAG_ENV not in env and ALLOW_ENV not in env, f'{phase} already carries candidate env')
        env[FLAG_ENV] = '1'
        env[ALLOW_ENV] = AA3E_HEAD
    expected = [f'source_files[{heads[0]}].sha256', f'source_files[{guards[0]}].sha256',
                'source_selection.production_head_sha', 'source_selection.guard_sha',
                'forbidden_environment'] + [
                    f'phases.{p}.service_environment.{key}' for p in PHASES
                    for key in (FLAG_ENV, ALLOW_ENV)]
    return m, sorted(expected), {'head_index': heads[0], 'guard_index': guards[0]}


def verify(base: dict, candidate: dict) -> None:
    _, expected, _ = derive(base)
    changed = leaf_diff(base, candidate)
    require(sorted(changed) == expected,
            f'candidate manifest drift outside the whitelist: {sorted(changed)}')
    head = [item for item in candidate['source_files'] if item['path'].endswith(HEAD_SUFFIX)][0]
    guard = [item for item in candidate['source_files'] if item['path'].endswith(GUARD_SUFFIX)][0]
    require(head['sha256'] == AA3E_HEAD and guard['sha256'] == GUARD_POST, 'candidate source binding wrong')
    require(candidate['source_selection']['production_head_sha'] == AA3E_HEAD, 'candidate head selection wrong')
    require(candidate['source_selection']['guard_sha'] == GUARD_POST, 'candidate guard selection wrong')
    require(FLAG_ENV not in candidate['forbidden_environment'], 'fused flag still forbidden')
    for phase in PHASES:
        env = candidate['phases'][phase]['service_environment']
        require(env.get(FLAG_ENV) == '1' and env.get(ALLOW_ENV) == AA3E_HEAD,
                f'{phase} candidate environment wrong')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--base-sha', default=BASE_SHA)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--diff-out', type=Path)
    ap.add_argument('--check', action='store_true', help='verify an existing candidate file instead of writing')
    a = ap.parse_args()
    require(sha(a.base) == a.base_sha, 'base manifest SHA mismatch')
    base = json.loads(a.base.read_text())
    if a.check:
        require(a.out.is_file(), 'candidate manifest missing')
        candidate = json.loads(a.out.read_text())
        verify(base, candidate)
        print(json.dumps({'status': 'verified', 'candidate_sha256': sha(a.out)}, indent=1))
        return 0
    require(not a.out.exists(), 'candidate manifest already exists; choose a new path')
    candidate, expected, indexes = derive(base)
    verify(base, candidate)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + '\n')
    changed = leaf_diff(base, candidate)
    def shown(value):
        return '<absent>' if isinstance(value, _Missing) else value
    report = {'schema': 'candidate-manifest-diff/1',
              'base_sha256': a.base_sha, 'candidate_sha256': sha(a.out),
              'indexes': indexes, 'changed': sorted(changed),
              'values': {k: {'before': shown(v[0]), 'after': shown(v[1])}
                         for k, v in sorted(changed.items())}}
    if a.diff_out:
        a.diff_out.parent.mkdir(parents=True, exist_ok=True)
        a.diff_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'written', 'candidate_sha256': report['candidate_sha256'],
                      'changed': report['changed']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
