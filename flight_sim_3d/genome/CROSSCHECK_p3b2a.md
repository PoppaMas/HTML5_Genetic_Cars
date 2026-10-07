# CROSSCHECK: phase3_b2a vs FD B2a / B1 r1 (2026-10-06 ~21:35 PT)

## Pins
| aircraft | full_a1_b2a |
|---|---|
| c172x | `full_a1_b2a:flexv2b2a:847bed9b` |
| T38 | `full_a1_b2a:flexv2b2a:b635a51d` |
| 737 | `full_a1_b2a:flexv2b2a:5d5a8f17` |
| f16 | `full_a1_b2a:flexv2b2a:07b08913` |

B1 r1 strings unchanged. Genome evaluator: `via='fd'` (`flexeval_b2.evaluate`) — ER `FIDELITIES` has no `full_a1_b2a` yet.

## Gene layout
- Controller 8 + structure_v2 12 + shape_b2 **9 active** = **29**
- `locked_genes`: `wing_tc_root_scale`, `wing_tc_tip_ratio` (injected as FD defaults 1.0; excluded from GA vector)
- Dihedral 0…+3 (0…+2 on 737); tc_root floors raised per FD B2a r0 review

## V1: B2a at defaults ≡ B1 r1
See `runs/p3b2a_verify_c172x.json`. Baseline tip-verify cost **0.2419400885448951** bit-identical (cost, 24 terms, per-scenario, tip BM) across empty-dict / full-defaults / encoded-identity-u / lock-injected encodings. Soft tip and p3b1 fly-shapes likewise bit-identical vs same-process B1. `model_version` / fidelity labels differ by design.

## Operators
`block_ops.py` maps `shape_b2` → shape block; phase3_b1 / phase3_b1_x golden paths unchanged. Uniform crossover draws `rng.random(2+n_s)` = 11 for B2a.
