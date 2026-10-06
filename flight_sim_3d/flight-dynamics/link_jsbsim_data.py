#!/usr/bin/env python3
"""Link jsbsim_root/engine and jsbsim_root/systems to the installed jsbsim package data. Run once after cloning.

jsbsim_root/aircraft/ holds Flight Dynamics' patched aircraft copies (in the repo). Engines and systems come
unmodified from the jsbsim pip package, as flexwing.prepare_aircraft sets them up. The links are machine-specific,
so they are not committed (see ../.gitignore). Usage (from flight_sim_3d/):

    python flight-dynamics/link_jsbsim_data.py
"""
import os
import shutil
import sys

import jsbsim

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jsbsim_root")
SRC = os.path.dirname(jsbsim.__file__)


def main() -> int:
    for d in ("engine", "systems"):
        link, target = os.path.join(ROOT, d), os.path.join(SRC, d)
        if not os.path.isdir(target):
            print(f"ERROR: {target} not found in the installed jsbsim package", file=sys.stderr)
            return 1
        if os.path.islink(link) and os.path.realpath(link) != os.path.realpath(target):
            os.unlink(link)  # stale/broken link from another machine or jsbsim install
        if os.path.exists(link):
            print(f"ok      {link} -> {os.path.realpath(link)}")
            continue
        try:
            os.symlink(target, link)
            print(f"linked  {link} -> {target}")
        except OSError:  # e.g. Windows without symlink rights: copy instead (read-only data)
            shutil.copytree(target, link)
            print(f"copied  {target} -> {link}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
