# Frozen Flight Dynamics copy for P3-B2a (`full_a1_b2a`)

Keep this folder. It is needed to re-fly P3-B2a runs exactly (`evolution/runs/phase3b2a-smoke-s1`,
`evolution/runs/phase3b2a-smoke-detfix-s1` and any run whose `run.json` `fd_dir` points here).

ER copied it from `flight-dynamics/` right after FD signed off B2a (2026-10-06, about 21:10 PT). The team's copy is the
full FD tree. This repo ships only the minimum Evolution needs, byte-identical to that copy, the same way as
`_fd_pin_p3b1r1/`. `model_version` hashes the code bytes and the aircraft files, so do not edit them:

* the 9 A1/B1 r1 files in `v2_results/FROZEN_A1_B1r1.md5` plus `planform_b2.py`, `flexbody_b2.py`, `flexeval_b2.py`
  (`v2_results/FROZEN_B2a.md5`)
* `v2_results/model_versions_post_{mass,p25,p3a1,p3b1,p3b1r1,p3b2a}.json`, `p3b2a_energy_ref.json` (frozen per-aircraft
  energy reference read by `flexeval_b2`) and both FROZEN md5 files
* `jsbsim_root/aircraft/`, `jsbsim_root_v2/aircraft/` and `jsbsim_root_v2b2/aircraft/`. The `engine/` and `systems/`
  links come from `python flight-dynamics/link_jsbsim_data.py`.

At push time, every frozen file here is byte-identical to the shipped `flight-dynamics/`.

Pins (`v2_results/model_versions_post_p3b2a.json`), `full_a1_b2a`: c172x `flexv2b2a:847bed9b`, T38 `b635a51d`,
737 `5d5a8f17`, f16 `07b08913`. All older fidelities are byte-identical to the post-P3-B1 r1 pins.

```bash
cd flight_sim_3d
EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a python -m evolution.batch --config evolution/configs/phase3b2a_smoke.json --model-versions
EVOLUTION_FD_DIR=evolution/_fd_pin_p3b2a python -m evolution.batch --config evolution/configs/phase3b2a_smoke.json --run-id p3b2a-smoke-repro-s1
(cd evolution/_fd_pin_p3b2a && grep -hv '^#\|^frozen' v2_results/FROZEN_A1_B1r1.md5 v2_results/FROZEN_B2a.md5 | md5sum -c -)
```
