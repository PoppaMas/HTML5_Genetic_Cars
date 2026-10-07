# Frozen Flight Dynamics copy for P3-B1 r1 (`full_a1_b1`)

Keep this folder. It is needed to re-fly P3-B1 r1 runs exactly (`evolution/runs/phase3b1r1-smoke-s1` and any run
whose `run.json` `fd_dir` / `fd_pin.json` points here).

ER copied it from `flight-dynamics/` on 2026-10-06 at about 18:50 PT, right after FD froze B1 r1 (18:47 MST). The team's
copy is the full FD tree. This repo ships only the minimum Evolution needs, byte-identical to that copy, the same way as
`_fd_pin_post_mass/`. `model_version` hashes the code bytes and the aircraft files, so do not edit them:

* the 9 files listed in `v2_results/FROZEN_A1_B1r1.md5`: `flexbody.py`, `flexwing.py`, `flexeval.py`, `coupled_sim.py`,
  `flexbody_a1.py`, `flexeval_a1.py`, `planform_b1.py`, `flexbody_b1.py` and `flexeval_b1.py`
* `v2_results/model_versions_post_{p25,p3a1,p3b1,p3b1r1}.json` and `FROZEN_A1_B1r1.md5`
* `jsbsim_root/aircraft/` and `jsbsim_root_v2/aircraft/`. The `engine/` and `systems/` links come from
  `python flight-dynamics/link_jsbsim_data.py`.

At push time, every frozen file here is byte-identical to the shipped `flight-dynamics/`.

Pins (`v2_results/model_versions_post_p3b1r1.json`):

* `full_a1_b1`: c172x `flexv2b1:56ee798e`, T38 `7e871977`, 737 `6523753c`, f16 `617078a9`
* `rigid` / `reduced` / `full` use the post-P2.5 pins, and `full_a1` uses the post-P3-A1 pins.

The r0 file (`model_versions_post_p3b1.json`) is superseded but kept unchanged.

```bash
cd flight_sim_3d
EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1 python -m evolution.batch --config evolution/configs/phase3b1_smoke.json --model-versions
EVOLUTION_FD_DIR=evolution/_fd_pin_p3b1r1 python -m evolution.batch --config evolution/configs/phase3b1_smoke.json --run-id p3b1r1-smoke-repro-s1
# verify the frozen bytes
(cd evolution/_fd_pin_p3b1r1 && grep -v '^frozen' v2_results/FROZEN_A1_B1r1.md5 | md5sum -c -)
```
