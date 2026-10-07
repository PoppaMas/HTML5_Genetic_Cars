# Frozen Flight Dynamics copy for the Phase 2 pilot seeds (post-mass, pre-P2.5)

The Phase 2 pilot runs `evolution/runs/phase2-pilot-s{1,2,3}` (and `phase2-smoke-s1`, the bench runs) are pinned to FD's
**post-mass** model versions (`v2_results/model_versions_post_mass.json`, e.g. `full:flexv2:11df8fe4` for c172x).
FD's P2.5 (2026-10-06 ~13:55 MST) changed `flexbody.py`, so the current `flight-dynamics/` produces different
`model_version` strings and Evolution refuses those configs. Seed 3 was finished from this copy.

The files are the minimum Evolution needs, byte-identical to the team's frozen copy (`model_version` hashes the code
bytes and the aircraft files, so do not edit them): `flexbody.py` (pre-P2.5), `flexeval.py`, `flexwing.py`,
`coupled_sim.py`, `jsbsim_root/aircraft/`, `jsbsim_root_v2/aircraft/`. Its `engine/`/`systems/` links come from
`python flight-dynamics/link_jsbsim_data.py`.

```bash
cd flight_sim_3d
EVOLUTION_FD_DIR=evolution/_fd_pin_post_mass python -m evolution.batch --config evolution/configs/phase2_pilot_s3.json --model-versions
EVOLUTION_FD_DIR=evolution/_fd_pin_post_mass python -m evolution.batch --config evolution/configs/phase2_smoke.json --run-id phase2-smoke-repro-s1
```
