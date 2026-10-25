"""
scripts/check_config.py

List config blocks that are not yet decided

usage: python scripts/check_config.py configs/primary.yaml
"""

import sys, yaml

c = yaml.safe_load(open(sys.argv[1]))
rows = list(c.items()) if isinstance(c, dict) else []

for k, v in rows:

    if k == "axes":
        for a in v:
            if a.get("_status") not in ("acked", "locked"): print(f"axes/{a['name']:14s} {a.get('_status')}")

    elif isinstance(v, dict) and v.get("_status") not in ("acked", "locked"):
        print(f"{k:20s} {v.get('_status')}  {v.get('_ref','')}")
