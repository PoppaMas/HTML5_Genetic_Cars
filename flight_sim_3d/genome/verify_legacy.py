#!/usr/bin/env python3
"""Compare two evolve.py output dirs (original vs adapter): CSV minus timing, JSON minus config.out.

    python verify_legacy.py runs/equiv_original runs/equiv_adapter
Exit code 0 iff identical.
"""
import csv
import json
import sys


def load(d):
    rows = list(csv.DictReader(open(f"{d}/fitness_history.csv")))
    for r in rows:
        r.pop("elapsed_s")
    js = json.load(open(f"{d}/best_gains.json"))
    js["config"].pop("out", None)
    js["config"].pop("workers", None)  # worker count does not affect results
    return rows, js


def main(a, b):
    ra, ja = load(a)
    rb, jb = load(b)
    ok = True
    if ra != rb:
        ok = False
        for i, (x, y) in enumerate(zip(ra, rb)):
            if x != y:
                print(f"CSV row {i} differs:\n  {x}\n  {y}")
                break
        if len(ra) != len(rb):
            print(f"CSV length {len(ra)} vs {len(rb)}")
    for k in sorted(set(ja) | set(jb)):
        if json.dumps(ja.get(k), sort_keys=True) != json.dumps(jb.get(k), sort_keys=True):  # NaN-safe
            ok = False
            print(f"JSON key {k!r} differs")
    print("IDENTICAL" if ok else "DIFFERENT", f"({len(ra)} generations compared)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
