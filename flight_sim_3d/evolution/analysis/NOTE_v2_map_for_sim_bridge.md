# Note for Sim Bridge — evolution flex_state /3 + v2_map 2.0.0

Wired 2026-10-06 (~09:40 PT). evolution imports `sim-bridge/sim_bridge/v2_map.py` **read-only** (file load; no
`sim_bridge` package import; stdlib only on that side).

## Schema versions to branch on
- Trajectory: `ga-flightsim-traj/2` (old Phase-2 pilot files remain `/1`; `validate_traj` accepts both).
- Flex state: `evolution-flex-state/3`.
- v2_map: `V2_MAP_VERSION = "2.0.0"`, structure block carries `structure.v2_map`.

## Channel layout
| fidelity | `wingR` / `wingL` | modal 9-node | empennage / fuselage / scalars |
|---|---|---|---|
| **full** | FD FE nodes via `map_v2_record(..., nodes=node_values)` — dof `dz`, `dx`, `twist` | `wingR_modal.*` / `wingL_modal.*` (`dy=0`) | `htail` / `vtail` / `fuselage` + `struct.*` |
| **reduced** | 9-node modal (unchanged names) | — | not emitted |

Pass `telemetry[i]['nodes']` (fd-flexbody-nodes/1) into `map_v2_record` / `nodes_frame` for exact shapes; without
nodes the map estimates from tip scalars and flags `estimated`.

## Public FlexState (no private helpers needed)
```python
fs.fd_model          # FlexBodyModel | FlexWing
fs.nodes()           # {body: {field: [float]}} or None
fs.node_layout()     # FD layout list or None
fs.v2_geometry       # v2_map geometry or None
fs.v2_map_version    # "2.0.0" or None
fs.channels() / fs.structure / fs.as_dict()
```
Builder: `evolution.fidelity.make_fd_model(profile_d, struct, fidelity)`.
`_aircraft_entry` / `_struct_obj` still exist for the batch path until you switch.

## evaluate pin
`evolution.eval.evaluate(..., pin="<model_version>")` raises `RuntimeError` on mismatch.
Batch `pin_model_version` + cache guard are unchanged. `ladder_model_version` stays `{fidelity: mv}`.
