#!/usr/bin/env python3
"""Execute a JSON argv list without a shell (no split/eval)."""
import json, subprocess, sys
argv = json.loads(sys.argv[1])
if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
    print("invalid window argv", file=sys.stderr); sys.exit(2)
p = subprocess.run(argv)
sys.exit(p.returncode)
