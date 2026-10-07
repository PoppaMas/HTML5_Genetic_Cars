# STATUS v2_map — DONE 09:44 PT

## Result
`$PY -m pytest evolution/tests -q`: **107 passed** in 95.5 s (09:42:52–09:44:28 PT, load 1.08→2.43). Log: `logs/pytest_v2_map.log`.
No Phase 2 re-runs. Local only; read-only folders untouched.

## What changed
| area | change |
|---|---|
| `fidelity.py` | `v2_map_mod()`, schema `/3`, FlexState public API, SBHook wires `map_v2_record`/`structure_block` with live `node_values`; modal → `wing*_modal` at full; `make_fd_model` |
| `trajectory.py` | schema `ga-flightsim-traj/2`; `dx` + `struct.*` in `channel_doc` |
| `validate_traj.py` | accepts `/1` and `/2` |
| `eval.py` | `evaluate(..., pin=)` raises on model_version mismatch |
| `tests/test_v2_map.py` | 7 new tests; recorder/trajectory asserts updated |
| docs | README flex_state section; `analysis/NOTE_v2_map_for_sim_bridge.md` |

## Schema strings
- `evolution-flex-state/3`
- `ga-flightsim-traj/2`
- v2_map `V2_MAP_VERSION = "2.0.0"`

## Public FlexState API
`.fd_model`, `.nodes()`, `.node_layout()`, `.v2_geometry`, `.v2_map_version`, `.channels()`, `.structure`, `.as_dict()`;
plus `fidelity.make_fd_model(profile_d, struct, fidelity)`. Private helpers kept.

## Unchanged (as promised)
- `ladder_model_version` = `{fidelity: model_version}` dict
- batch/cache `pin_model_version` enforcement
- reduced modal `wingR`/`wingL` names
