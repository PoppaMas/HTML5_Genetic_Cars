# B2 viewer export spec — FINAL for B2a (FD sign-off; frozen files in flight-dynamics/v2_results/FROZEN_B2a.md5)

Status: FD signed off B2a; this note was `b2_fd_spec_provisional.md` (provisional text from 21:13 PT 2026-10-06) and is
now final for B2a. B2b (area_scale / aspect_scale) is still pending. Verified 2026-10-06 against the frozen files
(`md5sum -c v2_results/FROZEN_B2a.md5`: 9/9 OK) and INTERFACE_v2.md §15.6.

Code path (read-only): /workspace/flight-sim-team/flight-dynamics/ `flexbody_b2.node_layout_b2(mdl, rp_offset_body_ft=None)`
or `FlexBodyModelB2(...).node_layout(rp)`. It wraps node_layout_b1 and returns the same component dicts.
If all B2 genes are at default, the output is exactly node_layout_b1 with no new keys, so treat the B2 keys as OPTIONAL.
API: INTERFACE_v2.md §15.6. Fixture: c172x via flexeval_b2 / planform_b2, e.g. dihedral +3, tc_root 1.2, camber_root +1.
Fixture regenerated from the frozen files: `tools/make_b2a_fixture.py` -> tests/fixtures/b2a_c172x_node_layout.json + data/b2a_fixture/.

Fields (wingR / wingL only; present only when not at defaults; all B1 r1 keys kept):
- axis_nodes_body_ft, le_nodes_body_ft, te_nodes_body_ft now include the dihedral delta: z += -s*tan(dGamma), with s = the FE beam
  station from the wing root. The beam station is spanwise (|y| - |y0|), not the length along a swept EA, so this matches
  INTERFACE_v2 §15.6 `z = -(y - y0) tan dGamma` (checked on 737 / T38 / c172x: max error 5e-7 ft; along the swept EA would be +9 % on the 737).
  Baseline dihedral was never in the nodes. FD exports feet; the traj `_m` keys are converted downstream.
- tc_local: per-node absolute t/c (dimensionless, e.g. 0.12)
- camber_meq_pct_local: per-node absolute max camber, % chord (baseline + delta)
- dihedral_delta_deg, dihedral_baseline_deg: scalars in degrees (the baseline is notional and not drawn into the nodes)
- section_baseline: string, e.g. "NACA 2412"
- chord_ft, geometric_twist_deg per node, as in B1 r1
- Thickness and camber are metadata only (no airfoil surface). Build the section display from chord + t/c + camber ourselves.
- B2b (tomorrow): area_scale / aspect_scale + span-scaled stations. Plan for them, but they won't appear in B2a.

Signs: body FRD. Dihedral + = tip up, so z becomes more negative toward both tips (symmetric L/R).
B2a range 0..+3 deg (737 max +2); anhedral disabled.

## Verified against the frozen code (sim-bridge, 2026-10-06)
- node_layout_b2 with B2 at default == node_layout_b1 exactly (no new keys).
- B2 non-default: x / y of axis / LE / TE unchanged; z shifted as above on BOTH wings (symmetric); tc_local /
  camber_meq_pct_local / dihedral_* / section_baseline on wingR and wingL. Any non-default B2 gene emits all of them
  (e.g. dihedral only still writes tc_local / camber at their baseline values).
- c172x: section_baseline "NACA 2412" but camber_meq_pct_local at default is 1.8 (%c), not 2 — "meq" is FD's
  equivalent camber, so the viewer shows the number FD exports, not the name's digits. dihedral_baseline_deg 1.73.
- FD exports ft / deg (chord_ft, geometric_twist_deg, *_nodes_body_ft). ER's `fidelity.b1_node_fields` converts only
  the B1 r1 keys (chord_m, geometric_twist_rad, le/te_nodes_body_m). The B2 keys must be passed through, and the
  dihedral z must reach axis_nodes_body_m. The fixture does this (same conversion rules, dimensionless keys unchanged).

## Viewer (sim-bridge)
- Every B2 key is optional. No key present, or all at default (dihedral 0, area / aspect 1, B2 genes at default) ->
  B1 r1 exactly.
- Dihedral: node z relative to the root node (EA) is added to the procedural wing. It is not recomputed from
  dihedral_delta_deg, and the procedural baseline dihedral is kept. HUD: `B2a dihedral Δ +3.0° (baseline 1.7°) · NACA 2412`.
- Section: NACA-4-style sketch (root solid, tip dashed, true relative chord) from chord_m + tc_local +
  camber_meq_pct_local, with the camber position from the section name (else 0.4). Metadata only.
- B2b ready: area_scale / aspect_scale -> stage "B2b", HUD size line, wing span from FD's node tip y (span-scaled
  stations) instead of the procedural span.
- On-page note when B2 fields are present: "B2a: dihedral baked into FD node layout; t/c and camber shown as section metadata."
