#!/usr/bin/env python3
"""Check that every hook point in touchpoints.toml is still in the tree (run after a rebase onto a new upstream)."""
import pathlib, sys, tomllib

root = pathlib.Path(__file__).resolve().parent.parent
hooks = tomllib.loads((root / "touchpoints.toml").read_text())["hook"]
missing = 0
for h in hooks:
    path = root / h["file"]
    if not path.exists() or h["probe"] not in path.read_text(errors="replace"):
        print(f"MISSING  [{h['feature']}] {h['file']}: {h['probe']}" + (f"  ({h['note']})" if "note" in h else ""))
        missing += 1
print(f"{len(hooks) - missing}/{len(hooks)} hook points present")
sys.exit(1 if missing else 0)
