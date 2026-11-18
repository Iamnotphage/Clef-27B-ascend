#!/usr/bin/env python3
"""5c external-vendor hook: validate immutable system API; installation is empty.

The Grouped29 adapter is a staged payload. No vendor, loader path, system
library or service source is modified by check/install/revert.
"""
import argparse
import json
from pathlib import Path
from window_common import check_head, frozen, staged_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for mode in ('check', 'install', 'revert'):
        modes.add_argument('--' + mode, action='store_true')
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--expect-payload-sha', required=True)
    a = p.parse_args()
    try:
        stage, run = staged_paths(__file__, a.run_dir)
        check_head(a.tree)
        identity = frozen(stage, a.expect_payload_sha)
        result = {'schema': 'grouped29-immutable-hook/1', 'pass': True,
                  'action': next(k for k in ('check', 'install', 'revert') if getattr(a, k)),
                  'system_mutations': [], 'vendor_installation': None,
                  'adapter_sha256': identity['adapter']['sha256']}
    except Exception as exc:
        result = {'schema': 'grouped29-immutable-hook/1', 'pass': False, 'error': str(exc)}
    print(json.dumps(result))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
