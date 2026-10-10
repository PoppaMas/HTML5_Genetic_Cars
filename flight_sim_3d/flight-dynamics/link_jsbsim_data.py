#!/usr/bin/env python3
"""Link <root>/engine and <root>/systems to the installed jsbsim package data, for both model roots
(jsbsim_root: rigid / flex v1, jsbsim_root_v2: flex v2, jsbsim_root_v2b2: P3-B2a), in flight-dynamics/ and in the frozen FD
copies evolution/_fd_pin_post_mass/, _fd_pin_p3b1r1/, _fd_pin_p3b2a/ and _fd_pin_p4cs/ if present. Run once after cloning.

<root>/aircraft/ holds Flight Dynamics' patched aircraft copies (in the repo). Engines and systems come
unmodified from the jsbsim pip package, as flexwing.prepare_aircraft sets them up. The links are machine-specific,
so they are not committed (see ../.gitignore). Usage (from flight_sim_3d/):

    python flight-dynamics/link_jsbsim_data.py
"""
import os
import shutil
import sys

import jsbsim

HERE = os.path.dirname(os.path.abspath(__file__))
# FD's model roots, plus the frozen FD copies selected with EVOLUTION_FD_DIR: pre-P2.5 (Phase 2 pilot seeds), P3-B1 r1, P3-B2a
PINS = [os.path.join(os.path.dirname(HERE), "evolution", p) for p in ("_fd_pin_post_mass", "_fd_pin_p3b1r1", "_fd_pin_p3b2a", "_fd_pin_p4cs")]
ROOTS = [os.path.join(d, r) for d in (HERE, *PINS) for r in ("jsbsim_root", "jsbsim_root_v2", "jsbsim_root_v2b2")]
SRC = os.path.dirname(jsbsim.__file__)


def main() -> int:
    for root, d in [(r, d) for r in ROOTS if os.path.isdir(r) for d in ("engine", "systems")]:
        link, target = os.path.join(root, d), os.path.join(SRC, d)
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
