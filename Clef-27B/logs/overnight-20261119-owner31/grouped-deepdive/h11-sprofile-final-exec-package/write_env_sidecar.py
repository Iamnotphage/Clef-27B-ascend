#!/usr/bin/env python3
"""Write a fresh env sidecar from the captured identity and selected env values."""
import json, os, sys
from pathlib import Path
pid = Path(sys.argv[1]); out = Path(sys.argv[2])
ident = json.loads(pid.read_text())
data = {"schema": "h11-sprofile-env-sidecar/1", "owner": os.environ.get("CLEF_WINDOW_OWNER", ""),
        "identity": ident, "env": {k: os.environ.get(k) for k in
        ("ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_PROCESS_LOG_PATH", "CLEF_WINDOW_OWNER")}}
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(json.dumps(data, indent=1) + "\n")
os.replace(tmp, out)
