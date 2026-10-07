"""Regenerate evolution/configs/phase2_*.json from configs/phase1_hdg.json (v4 controller).
Run from the team root:  $PY evolution/analysis/make_phase2_configs.py [--placeholder] [--clip] [--p25 | --p3a1 | --p3b1 [--r0]] [--seed N]
--clip: also set Genome's interim tail/fuselage mass-credit clip (flex_mass_credit_clip [ht, vt, fus]). Default OFF,
as Genome's phase2_flex since 08:15 PT (FD's section 12 fix replaced it; Genome keeps it for A/B only).
Pins: default flight-dynamics/v2_results/model_versions_post_mass.json (EVOLUTION_FD_DIR honoured).
--p25: pins from model_versions_post_p25.json and write ONLY phase2_smoke_p25.json + phase2_pilot_p25.json
(does not overwrite post-mass pilot/smoke/bench). --placeholder writes PENDING-FD-NEW-MODEL-VERSION.
--p3a1: FD's P3-A1 fidelity full_a1 (INTERFACE_v2 section 13). Writes ONLY phase3a1_smoke.json (= phase2_smoke_p25.json
with fidelity full_a1: 16 x 5, c172x/T38/737, seeded sigma 0.10, asymmetric off) and phase3a1_pilot.json (64 x 60,
rigid->full_a1, seeds as the P2.5 pilot). Pins: full_a1 from model_versions_post_p3a1.json, rigid from
model_versions_post_p25.json (the smoke has no rigid rung, so it pins full_a1 only). Nothing else is overwritten.
--p3b1: FD's P3-B1 fidelity full_a1_b1 (INTERFACE_v2 section 14) with genome_kind phase3_b1 (controller | 12 structure
genes | FD's 6 shape genes). Writes ONLY phase3b1_smoke.json (= phase3a1_smoke.json with fidelity full_a1_b1 + kind
phase3_b1: 16 x 5, c172x/T38/737, same seeds / scenarios / sigma 0.10) and phase3b1_pilot.json (64 x 60, rigid->full_a1_b1,
seeds as phase3a1_pilot). Pins: full_a1_b1 from model_versions_post_p3b1r1.json (FD B1 r1, frozen 18:47 PT; = the
default since r1), rigid from model_versions_post_p25.json.
--p3b1 --r0: the superseded r0 pins (model_versions_post_p3b1.json) -> phase3b1_smoke_r0.json / phase3b1_pilot_r0.json
(kept for the record / replay of phase3b1-smoke-s1; do not pilot on them)."""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "configs")
PH = "PENDING-FD-NEW-MODEL-VERSION"
ACS = [("c172x", "phase1_c172x"), ("T38", "phase1_T38"), ("737", "phase1_737")]


def profiles(hdg):
    out = {}
    for name, p in ACS:
        d = copy.deepcopy(hdg["profiles"][p])
        d["_why"] = ("phase1_hdg.json (v4 controller = Genome phase2_flex's phase1_v4 block) with v5's ki_alt upper bound 0.5"
                     + ("; Genome's interim tail/fuselage mass-credit clip ON (A/B)" if CLIP else ""))
        d["aircraft_root"] = "flight-dynamics/jsbsim_root"   # team-relative; follows EVOLUTION_FD_DIR; full uses <root>_v2
        d["gain_bounds"]["ki_alt"][1] = 0.5
        if CLIP:
            d["flex_mass_credit_clip"] = ["ht", "vt", "fus"]
        out["phase2_" + name] = d
    return out


_FD_ROOT = os.environ.get("EVOLUTION_FD_DIR") or os.path.join(HERE, "..", "..", "flight-dynamics")
P25 = "--p25" in sys.argv[1:]
P3A1 = "--p3a1" in sys.argv[1:]
P3B1 = "--p3b1" in sys.argv[1:]
if P25 + P3A1 + P3B1 > 1:
    sys.exit("--p25, --p3a1 and --p3b1 are exclusive")
PIN_NAME = "model_versions_post_p25.json" if (P25 or P3A1 or P3B1) else "model_versions_post_mass.json"
A1_PIN_NAME = "model_versions_post_p3a1.json"
B1_R0 = P3B1 and "--r0" in sys.argv[1:]
B1_PIN_NAME = "model_versions_post_p3b1.json" if B1_R0 else "model_versions_post_p3b1r1.json"
B1_REV = "r0 (SUPERSEDED by r1)" if B1_R0 else "r1"
B1_SUFFIX = "_r0" if B1_R0 else ""
PIN_FILE = os.path.join(_FD_ROOT, "v2_results", PIN_NAME)
PIN_SOURCE = f"flight-dynamics/v2_results/{PIN_NAME}"
FD_PINS = {}
CLIP = "--clip" in sys.argv[1:]


def pins(fids):
    return {n: {f: FD_PINS.get(n, {}).get(f, PH) for f in fids[n]} for n, _ in ACS}


def main():
    global FD_PINS
    if "--placeholder" not in sys.argv[1:] and os.path.exists(PIN_FILE):
        FD_PINS = json.load(open(PIN_FILE))
    hdg = json.load(open(os.path.join(CFG, "phase1_hdg.json")))
    common = {"seed": 1, "scenario_seed": 1, "scenarios": 3, "metrics": hdg["metrics"], "fidelity": "full",
              "struct_genes": True, "struct_asymmetric": False,
              "init": {"mode": "baseline", "sigma": 0.10, "blocks": ["struct"]},
              "fidelity_per_aircraft": {"c172x": {"reduced_gate": 0.9}, "T38": {"reduced_gate": 1.0},
                                        "737": {"reduced_gate": 1.0}},
              "viz": "off", "profiles": profiles(hdg),
              "_pin_source": PIN_SOURCE + " (Flight Dynamics, post mass-credit fix, 2026-10-06 08:13 PT); reduced pins are FD's "
                             "default-gate strings, accepted for the run's own gated string (batch.check_pins)",
              "aircraft": [{"name": n, "profile": "phase2_" + n} for n, _ in ACS]}
    pilot = {"_comment": "Phase 2 PILOT (do not launch before Genome confirms FD's 7 new sizing/peak terms are wired and "
             "the smoke results are reviewed). Genome phase2_flex: v4 controller (8 genes, ki_alt <= 0.5) + FD's 12 v2 struct genes = 20 genes, "
             "asymmetric off, generation 0 seeded at FD's baseline (sigma 0.10 normalized; Genome's preset uses 0.05), "
             "FD's own mass term (Genome's interim clip off, as phase2_flex since 08:15 PT). Ladders: c172x rigid->reduced->full; T38/737 rigid->full with >= 25% of "
             "the population + elites re-scored at full. Feasibility only from full.",
             **copy.deepcopy(common), "ga": {"pop_size": 64, "generations": 60},
             "multi_fidelity": {"enabled": True, "screen": "rigid", "top_k": 4, "min_full_frac": 0.25, "mid_k": None},
             # all three aircraft: rigid->full (>=25% + elites at full). c172x no longer screens reduced
             # (ladder-bench recommendation 2026-10-06). Pins only need full (reduced unused).
             "pin_model_version": pins({"c172x": ["full"], "T38": ["full"], "737": ["full"]})}
    smoke = {"_comment": "Phase 2 SMOKE: 16 x 5 per aircraft, every evaluation at full fidelity (no ladder), same genome / "
             "init / clip as phase2_pilot.json.",
             **copy.deepcopy(common), "ga": {"pop_size": 16, "generations": 5},
             "pin_model_version": pins({"c172x": ["full"], "T38": ["full"], "737": ["full"]})}
    bench = {**copy.deepcopy(common), "ga": {"pop_size": 32, "generations": 8},
             "cache": {"enabled": False, "path": "cache/evals.sqlite"}}
    b_rf = {"_comment": "Phase 2 ladder BENCHMARK, variant A: rigid->full on all three aircraft (>= 25% + elites at full). "
            "Compare with phase2_bench_rrf.json (same seed / budget / genome, rigid->reduced->full) using "
            "evolution/analysis/phase2_check.py --compare. Cache off so each variant pays for its own evaluations.",
            **copy.deepcopy(bench),
            "multi_fidelity": {"enabled": True, "screen": "rigid", "top_k": 4, "min_full_frac": 0.25, "mid_k": None},
            "pin_model_version": pins({"c172x": ["full"], "T38": ["full"], "737": ["full"]})}
    b_rrf = {"_comment": "Phase 2 ladder BENCHMARK, variant B: rigid->reduced->full on all three aircraft (>= 25% + elites "
             "at full). See phase2_bench_rf.json.", **copy.deepcopy(bench),
             "multi_fidelity": {"enabled": True, "screen": ["rigid", "reduced"], "top_k": 4, "min_full_frac": 0.25,
                                "mid_k": None},
             "pin_model_version": pins({"c172x": ["reduced", "full"], "T38": ["reduced", "full"], "737": ["reduced", "full"]})}
    # distinct GA seeds per aircraft (the smoke run, one shared seed, drew identical generation-0 genomes on all three)
    seeds = {"c172x": 1, "T38": 2, "737": 3}
    for d in (pilot, b_rf, b_rrf):
        d["aircraft"] = [{**x, "seed": seeds[x["name"]]} for x in d["aircraft"]]
    c_full = {"_comment": "Phase 2 c172x FULL-ONLY comparison run (32 x 15, every evaluation at full, no ladder; seed 1 = the "
              "pilot's c172x seed) for wall time / rank agreement against the pilot's rigid->reduced->full ladder.",
              **copy.deepcopy(common), "ga": {"pop_size": 32, "generations": 15},
              "aircraft": [{"name": "c172x", "profile": "phase2_c172x", "seed": 1}],
              "fidelity_per_aircraft": {"c172x": {"reduced_gate": 0.9}},
              "pin_model_version": {"c172x": pins({"c172x": ["full"], "T38": [], "737": []})["c172x"]}}
    if P3A1:
        write_p3a1(smoke, pilot)
        return
    if P3B1:
        write_p3b1(smoke, pilot)
        return
    if P25:
        # NEW configs only — do not overwrite post-mass phase2_pilot / smoke / bench.
        common["_pin_source"] = (PIN_SOURCE + " (Flight Dynamics P2.5, 2026-10-06: wing_nsm floor 1.0–1.25, "
                                 "TERM_KEYS=24 + J_wing_tip_bm_limit); reduced pins are FD's default-gate strings")
        smoke["_comment"] = ("Phase 2 SMOKE P2.5: 16 x 5 full-only, pins from model_versions_post_p25.json "
                             "(nsm floor 1.0, tip BM sizing term).")
        smoke["_pin_source"] = common["_pin_source"]
        pilot["_comment"] = ("Phase 2 PILOT P2.5: 64 x 60 rigid->full all three, pins from model_versions_post_p25.json "
                             "(nsm floor 1.0, J_wing_tip_bm_limit). Do not launch 64x60 until smoke reviewed.")
        pilot["_pin_source"] = common["_pin_source"]
        for fn, d in (("phase2_smoke_p25", smoke), ("phase2_pilot_p25", pilot)):
            with open(os.path.join(CFG, fn + ".json"), "w") as f:
                json.dump(d, f, indent=1)
                f.write("\n")
            print("wrote", fn)
        return
    for fn, d in (("phase2_c172x_full", c_full), ("phase2_pilot", pilot), ("phase2_smoke", smoke), ("phase2_bench_rf", b_rf), ("phase2_bench_rrf", b_rrf)):
        with open(os.path.join(CFG, fn + ".json"), "w") as f:
            json.dump(d, f, indent=1)
            f.write("\n")
        print("wrote", fn)
    # optional: --seed N -> phase2_pilot_sN.json with c172x=N, T38=N+1, 737=N+2, scenario_seed=N
    seed_args = [a for a in sys.argv[1:] if a.startswith("--seed")]
    if seed_args:
        raw = seed_args[0].split("=", 1)[1] if "=" in seed_args[0] else sys.argv[sys.argv.index("--seed") + 1]
        n = int(raw)
        d = copy.deepcopy(pilot)
        d["seed"] = n
        d["scenario_seed"] = n
        d["aircraft"] = [{"name": "c172x", "profile": "phase2_c172x", "seed": n},
                         {"name": "T38", "profile": "phase2_T38", "seed": n + 1},
                         {"name": "737", "profile": "phase2_737", "seed": n + 2}]
        d["_comment"] = (f"Phase 2 PILOT seed {n}: same as phase2_pilot.json but GA seeds c172x={n}, T38={n+1}, 737={n+2}, "
                         f"scenario_seed={n}. Ladder rigid->full on all three (no c172x reduced screen).")
        fn = f"phase2_pilot_s{n}"
        with open(os.path.join(CFG, fn + ".json"), "w") as f:
            json.dump(d, f, indent=1)
            f.write("\n")
        print("wrote", fn)


def write_p3b1(smoke, pilot):
    """phase3b1_smoke.json / phase3b1_pilot.json: the A1 smoke / pilot (= P2.5 smoke / pilot) with fidelity full_a1_b1
    and genome_kind phase3_b1 (FD P3-B1 shape block; Genome's operator spec in batch.SHAPE_OPS_DEFAULT). Only fidelity,
    genome_kind, pins and comments differ from what --p3a1 writes (checked in tests/test_p3b1.py)."""
    b1 = {} if "--placeholder" in sys.argv[1:] else json.load(open(os.path.join(_FD_ROOT, "v2_results", B1_PIN_NAME)))
    names = [n for n, _ in ACS]
    src = (f"full_a1_b1: flight-dynamics/v2_results/{B1_PIN_NAME} (Flight Dynamics P3-B1 {B1_REV}, 2026-10-06); rigid: "
           f"{PIN_SOURCE} (P2.5; byte-identical in post_p3b1 / post_p3b1r1)"
           + ("" if B1_R0 else "; frozen FD copy for exact replay: evolution/_fd_pin_p3b1r1 (EVOLUTION_FD_DIR)"))
    smoke = copy.deepcopy(smoke)
    smoke["_comment"] = (f"Phase 3 B1 {B1_REV} SMOKE: = phase3a1_smoke.json (16 x 5 single-fidelity, c172x/T38/737, seeded sigma 0.10, "
                         "asymmetric off, same seeds / scenarios) with fidelity full_a1_b1 (FD P3-B1: A1 host + 6 planform "
                         "shape genes) and genome_kind phase3_b1 (controller 8 | structure 12 | shape 6 = 26 genes; shape gen-0 "
                         "around the identity planform, sigma 0.25 x half-range in FD's encoded space, log for the chord "
                         "tapers; whole-block crossover). Separate cache key space from full_a1.")
    smoke["_pin_source"] = src
    smoke["fidelity"] = "full_a1_b1"
    smoke["genome_kind"] = "phase3_b1"
    smoke["pin_model_version"] = {n: {"full_a1_b1": b1.get(n, {}).get("full_a1_b1", PH)} for n in names}
    pilot = copy.deepcopy(pilot)
    pilot["_comment"] = (f"Phase 3 B1 {B1_REV} PILOT: 64 x 60 rigid->full_a1_b1 on c172x/T38/737 (>= 25 % + elites re-scored at "
                         "full_a1_b1; the rigid screen ignores the shape genes), genome_kind phase3_b1, seeds as "
                         "phase3a1_pilot (GA c172x=1, T38=2, 737=3, scenario_seed 1). DO NOT LAUNCH without Corleone's "
                         "approval (gate: B1 smoke reviewed).")
    pilot["_pin_source"] = src
    pilot["fidelity"] = "full_a1_b1"
    pilot["genome_kind"] = "phase3_b1"
    pilot["pin_model_version"] = {n: {"rigid": FD_PINS.get(n, {}).get("rigid", PH),
                                      "full_a1_b1": b1.get(n, {}).get("full_a1_b1", PH)} for n in names}
    for fn, d in (("phase3b1_smoke" + B1_SUFFIX, smoke), ("phase3b1_pilot" + B1_SUFFIX, pilot)):
        with open(os.path.join(CFG, fn + ".json"), "w") as f:
            json.dump(d, f, indent=1)
            f.write("\n")
        print("wrote", fn)


def write_p3a1(smoke, pilot):
    """phase3a1_smoke.json / phase3a1_pilot.json: the P2.5 smoke / pilot with fidelity full_a1 (FD P3-A1). Only the
    fidelity, the pins and the comments differ from what --p25 writes (checked in tests/test_p3a1.py)."""
    a1 = {} if "--placeholder" in sys.argv[1:] else json.load(open(os.path.join(_FD_ROOT, "v2_results", A1_PIN_NAME)))
    names = [n for n, _ in ACS]
    src = (f"full_a1: flight-dynamics/v2_results/{A1_PIN_NAME} (Flight Dynamics P3-A1, 2026-10-06); rigid: {PIN_SOURCE} "
           "(P2.5; byte-identical in both files)")
    smoke = copy.deepcopy(smoke)
    smoke["_comment"] = ("Phase 3 A1 SMOKE: = phase2_smoke_p25.json (16 x 5 full-only, c172x/T38/737, seeded sigma 0.10, "
                         "asymmetric off) with fidelity full_a1 (FD P3-A1: 64-strip wings, 4b+3t+2ip, station-exact "
                         "J_wing_tip_bm_limit). Separate cache key space from full.")
    smoke["_pin_source"] = src
    smoke["fidelity"] = "full_a1"
    smoke["pin_model_version"] = {n: {"full_a1": a1.get(n, {}).get("full_a1", PH)} for n in names}
    pilot = copy.deepcopy(pilot)
    pilot["_comment"] = ("Phase 3 A1 PILOT: 64 x 60 rigid->full_a1 on c172x/T38/737 (>= 25 % + elites re-scored at full_a1), "
                         "seeds as phase2_pilot_p25 (GA c172x=1, T38=2, 737=3, scenario_seed 1). DO NOT LAUNCH without "
                         "approval (gate: A1 smoke reviewed).")
    pilot["_pin_source"] = src
    pilot["fidelity"] = "full_a1"
    pilot["pin_model_version"] = {n: {"rigid": FD_PINS.get(n, {}).get("rigid", PH),
                                      "full_a1": a1.get(n, {}).get("full_a1", PH)} for n in names}
    for fn, d in (("phase3a1_smoke", smoke), ("phase3a1_pilot", pilot)):
        with open(os.path.join(CFG, fn + ".json"), "w") as f:
            json.dump(d, f, indent=1)
            f.write("\n")
        print("wrote", fn)


if __name__ == "__main__":
    main()
