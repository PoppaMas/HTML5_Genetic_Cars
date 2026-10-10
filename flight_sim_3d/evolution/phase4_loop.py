"""Phase 4 ring-course GA loop (opt-in; only configs with "phase": "phase4_rings" reach this module).

    EVOLUTION_FD_DIR=evolution/_fd_pin_p4cs $PY -m evolution.batch --config evolution/configs/phase4_smoke.json
    ($PY -m evolution.phase4_loop --config ... is the same entry point)

Pieces (owners): FD plant flexeval_p4.fly_course (frozen pin evolution/_fd_pin_p4cs); Sim Bridge course + window/order/
time-limit rules sim_bridge.ring_course (make_course, ring_at, score_course, train/holdout seeds, frame helpers; called,
not duplicated); Genome chromosome mirrored in phase4_ga (bit cross-checked vs genome/runs/p4_operator_trace.json);
guidance law phase4_guidance (ER INTERIM until Genome ships its callable); ER scoring = this file + phase4_eval.

Determinism: one fly_course per process at a time (FD: not thread-safe); every course result depends only on
(pin, genes, model, stage, seed); aggregation is order-free (sorted + fsum); GA rng = default_rng([seed, model_tag]).
"""
from __future__ import annotations

import os as _os
# FD's flex plant (numpy inv/eigh) gives results that differ in the last ~10 digits with the BLAS thread count
# (measured 2026-10-07: T38 course cost 0.23497544924795072 multi-threaded vs 0.23497544943415924 single), so Phase 4 is
# single-threaded BLAS by contract; run() refuses otherwise. Setting it here only helps if numpy is not imported yet.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
FD_DIR = os.path.abspath(os.environ.get("EVOLUTION_FD_DIR", os.path.join(HERE, "_fd_pin_p4cs")))
SB_DIR = os.path.join(TEAM, "sim-bridge")
P4_FID = "full_a1_b2a_cs"
FID_ALIASES = {"full_a1_b2a_p4": P4_FID, P4_FID: P4_FID}
SCORING_VERSION = "er-p4-score/2"   # /2: + Mach overspeed, single-source FD limits
TRAJ_SCHEMA = "ga-flightsim-traj/2"
RINGS_PROFILE = {"c172x": "phase2_c172x", "T38": "phase2_T38", "737": "phase2_737"}

_M: Dict[str, object] = {}


def mods():
    """FD pin + Sim Bridge course module, imported once per process without bytecode."""
    if not _M:
        prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for p in (FD_DIR, SB_DIR, TEAM):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import flexeval as fe
            fe.PHASE1_CONFIG = os.path.join(HERE, "configs", "phase1.json")   # pin lives inside evolution/: fix its relative paths
            fe.default_profile.__defaults__ = (None, fe.PHASE1_CONFIG)
            fe.load_sim.__defaults__ = (os.path.join(HERE, "sim.py"),)
            sim = fe.load_sim()
            import flexeval_p4 as fp4
            from sim_bridge import ring_course as rc
        finally:
            sys.dont_write_bytecode = prev
        lim = json.load(open(os.path.join(FD_DIR, "v2_results", "p4_aircraft_limits.json")))["aircraft"]
        pins = json.load(open(os.path.join(FD_DIR, "v2_results", "model_versions_post_p4cs.json")))[P4_FID]
        _M.update(fe=fe, sim=sim, fp4=fp4, rc=rc, limits=lim, pins=pins)
    return _M


# ------------------------------------------------------------------------------------------------ limits (FD P4.10)
def course_limits(model: str, profile: Dict, v_ref_kcas: float) -> Dict:
    """v_ref = the course start KCAS (Sim Bridge AIRCRAFT v_ref_kts), not the profile's speed."""
    L = mods()["limits"][model]
    v_ref = float(v_ref_kcas)
    return {"alpha_stall_deg": float(L["alpha_stall_deg"]), "bank_course_deg": float(L["bank_course_deg"]),
            "v_max_kcas": min(1.25 * v_ref, float(L["v_max_kcas"])), "mach_max": L.get("mach_max"),
            "v_min_kcas": float(profile["min_kcas"]), "nz": tuple(L["n_profile"]), "n_inst": float(L["n_inst"]),
            "n_struct_limit": float(L["n_struct_limit"]),
            "attitude_fail": "none (fly_course applies no 45 deg fail; Phase 4 scores bank vs bank_course_deg)"}


# ------------------------------------------------------------------------------------------------ scoring
# J_carried = sum 1:1 of exactly these: FD's structural/aeroelastic subset of the 24 TERM_KEYS (every J_* key; the rigid
# track/effort/comfort/heading/hold are excluded: FD sets them 0 in fly_course and rings replace them), incl. the flutter
# margin term, + Evolution's J_energy and J_speed_guard (same rule as B2a, fidelity.ENERGY_W).
CARRIED_FD_KEYS = ("J_flutter_margin", "J_div_margin", "J_mass", "J_bm_rms", "J_bm_peak", "J_tip", "J_twist",
                   "J_reversal_margin", "J_tail_bm_peak", "J_fus_bm_peak", "J_smooth", "J_wing_bm_limit",
                   "J_wing_torque_limit", "J_wing_ip_limit", "J_wing_tip_bm_limit", "J_tail_bm_limit", "J_fus_bm_limit",
                   "J_wing_torque_peak", "J_wing_ip_peak")
CARRIED_KEYS = CARRIED_FD_KEYS + ("J_energy", "J_speed_guard")
def score_flight(r: Dict, course: Dict, model: str, lim: Dict, weights: Dict) -> Dict:
    from . import phase4_eval as PE
    rc = mods()["rc"]
    t = list(r.get("t") or [])
    M = int(course["M"])
    if len(t) >= 2:
        pos = rc.fd_pos_to_ned(r["pos"])
        sc = rc.score_course(t, pos, course)          # Sim Bridge owns window / strict order / time limit / rim (off)
    else:
        sc = {"gates": [], "passes": 0, "misses": 0, "M": M, "J_ring_miss": 1.0, "finished": False, "crash": False,
              "timed_out": False, "t_last": None, "nominal_time_s": course["nominal_time_s"]}
    passes = int(sc["passes"])
    gates = sc["gates"]
    flown_rho = [g["rho_norm"] for g in gates if g.get("rho_norm") is not None]
    miss_m = [g["miss_m"] for g in gates if g["result"] == "miss"]
    summ = {"passes": passes, "misses": int(sc["misses"]), "M": M, "pass_rate": passes / M, "finished": bool(sc["finished"]),
            "timed_out": bool(sc["timed_out"]), "crash": bool(sc["crash"]), "t_last": sc["t_last"],
            "nominal_time_s": float(course["nominal_time_s"]),
            "mean_rho_norm": (math.fsum(flown_rho) / len(flown_rho)) if flown_rho else None,
            "mean_rho_m": (math.fsum(g["rho_m"] for g in gates if g.get("rho_m") is not None) / len(flown_rho)) if flown_rho else None,
            "mean_miss_m": (math.fsum(miss_m) / len(miss_m)) if miss_m else None,
            "status": r["status"]}
    status = r["status"]
    hard = None
    if status != "ok" or r.get("struct_failed") or sc["crash"]:
        hard = {"ground": "ground_impact", "diverged": "divergence"}.get(status, status if status != "ok" else
                                                                         ("structural_failure" if r.get("struct_failed") else "rim_crash"))
    if hard:
        return {"cost": PE.HARD_FAIL_BASE + sc["J_ring_miss"], "hard_fail": hard, "pass_rate": passes / M,
                "terms": {k: 0.0 for k in PE.P4_TERM_KEYS}, "summary": summ, "gates": gates}
    tt = np.asarray(t, float)
    J = {"J_ring_miss": float(sc["J_ring_miss"])}
    # accuracy over all M rings: flown crossings use rho/r; order/time misses count as the cap (rho/r = 3)
    q = []
    for g in gates:
        if g.get("rho_norm") is None:
            q.append(PE._acc_q(3.0, False))
        else:
            q.append(PE._acc_q(g["rho_norm"], g["result"] == "pass"))
    q += [PE._acc_q(3.0, False)] * (M - len(gates))
    J["J_ring_acc"] = math.fsum(q) / M
    T_ref = float(course["nominal_time_s"])
    J["J_time"] = min(max(0.0, (sc["t_last"] - t[0]) / T_ref - 1.0), 1.0) if sc["finished"] else 1.0
    # surface use = ACTUAL surface positions (FD ctrl_surfaces <name>_deg at 120 Hz, JSBSim fcs/*-pos), never commands
    # (f16 FBW commands are rate/g demands); rates = d/dt of those positions (= FD <name>_rate_dps); rate-None (flap) skipped
    s120 = r.get("surfaces_120hz") or {}
    if "t_s" in s120:
        ts = np.asarray(s120["t_s"], float)
        act = {k: s120[f"{k}_deg"] for k in r["surface_limits"] if f"{k}_deg" in s120}
        J.update(PE.surface_terms(ts, act, r["surface_limits"]))
    else:
        J.update(PE.surface_terms(tt, r.get("surfaces") or {}, r["surface_limits"]))
    nz = np.asarray(r["nz"], float)
    lo, hi = lim["nz"]
    J["J_g"] = PE._tmean(tt, np.maximum(0, nz - hi) / hi + np.maximum(0, lo - nz) / abs(lo))
    phi = np.degrees(np.abs(np.asarray(r["att"], float)[:, 0]))
    B = lim["bank_course_deg"]
    J["J_bank"] = PE._tmean(tt, np.maximum(0, phi - B) / B)
    al = np.degrees(np.asarray(r["alpha"], float))
    A = lim["alpha_stall_deg"]
    J["J_aoa"] = PE._tmean(tt, np.maximum(0, al - A) / A)
    v = np.asarray(r["v_kcas"], float)
    J["J_overspeed"] = PE._tmean(tt, np.maximum(0, v - lim["v_max_kcas"]) / lim["v_max_kcas"])
    if lim.get("mach_max"):       # fly_course exports no Mach: M = TAS / a(ISA, altitude MSL = FD pos U), Sim Bridge's isa()
        mm = float(lim["mach_max"])
        h = np.clip(np.asarray(r["pos"], float)[:, 2], 0.0, 11000.0)
        a = np.array([rc.isa(float(x))[2] for x in h])
        mach = np.asarray(r["v_ms"], float) / a
        J["J_overspeed"] += PE._tmean(tt, np.maximum(0, mach - mm) / mm)
        summ["mach_max_flown"] = float(mach.max())
    J["J_underspeed"] = PE._tmean(tt, np.maximum(0, lim["v_min_kcas"] - v) / lim["v_min_kcas"])
    carried = {k: float((r.get("terms") or {}).get(k, 0.0)) for k in CARRIED_FD_KEYS}
    e = r.get("energy") or {}
    from .fidelity import ENERGY_W
    wE = ENERGY_W[model]
    carried["J_energy"] = wE * max(0.0, float(e.get("energy_drag_increment", 0.0) or 0.0))
    dfc, vt = e.get("speed_deficit_kts_mean"), e.get("v_target_kcas")
    carried["J_speed_guard"] = wE * 3.0 * max(0.0, float(dfc) - 2.0) / float(vt) if (dfc is not None and vt) else 0.0
    J["J_carried"] = math.fsum(carried[k] for k in sorted(carried))
    cost = math.fsum(weights[k] * J[k] for k in PE.P4_TERM_KEYS)
    summ["carried"] = carried
    return {"cost": float(cost), "hard_fail": None, "pass_rate": passes / M, "terms": J, "summary": summ, "gates": gates}


# ------------------------------------------------------------------------------------------------ one course
def profile_obj(cfg: Dict, model: str):
    m = mods()
    pd = {k: v for k, v in cfg["profiles"][RINGS_PROFILE_OF(cfg, model)].items() if not k.startswith("_")}
    return m["sim"].Profile.from_dict({**pd, "aircraft": model}), pd


def RINGS_PROFILE_OF(cfg, model):
    for a in cfg["aircraft"]:
        if a["name"] == model:
            return a["profile"]
    raise KeyError(model)


def genome_guidance():
    if "gg" not in _M:
        gdir = os.path.join(TEAM, "genome")
        prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            if gdir not in sys.path:
                sys.path.append(gdir)
            import p4_guidance as gg
        finally:
            sys.dont_write_bytecode = prev
        _M["gg"] = gg
    return _M["gg"]


def fly_one(cfg: Dict, model: str, u, stage: str, seed: int, record: bool = False, M: int = None) -> Dict:
    from . import phase4_ga as PG, phase4_guidance as GD
    m = mods()
    rc, fp4 = m["rc"], m["fp4"]
    prof, pd = profile_obj(cfg, model)
    course = rc.make_course(model, stage, int(seed), M=M)
    lim = course_limits(model, pd, course["start"]["kcas"])
    genes = PG.decode(np.asarray(u, float), model)
    log = [] if record else None
    if cfg.get("guidance", "genome") == "genome":       # Genome-owned law (genome/p4_guidance.make_guidance), default
        gd = genome_guidance().make_guidance(genes, model, course)
    else:                                              # "er_interim": evolution/phase4_guidance (tests / fallback only)
        gd = GD.make_guidance(genes, model, {**m["limits"][model], "bank_course_deg": lim["bank_course_deg"]}, rc, course)

    def guidance(s, gains, c, mdl):
        out = gd(s, gains, c, mdl)
        if log is not None:
            log.append((out["throttle"], out["elevator"], out["aileron"], out["rudder"]))
        return out
    fc = {"start": dict(course["start"]), "duration_s": float(course["time_limit_s"])}   # = genome/p4_guidance (SB: unresolved rings at t_limit = misses)
    if cfg.get("profile_mode") == "fd_default":       # cross-check mode: FD's Phase-1 default profile (as genome/p4_guidance)
        prof = None
    r = fp4.fly_course(prof, None, None, None, None, fc, fidelity=FID_ALIASES[cfg["fidelity"]], model=model,
                       guidance=guidance, sim=m["sim"], cs_mode="active", record_hz=30.0)
    res = score_flight(r, course, model, lim, cfg["weights"])
    res.update(model=model, stage=stage, seed=int(seed), model_version=r.get("model_version"))
    if record:
        res["_flight"] = r
        res["_cmd"] = log
        res["_course"] = course
        res["_genes"] = genes
    return res


def cache_key(cfg: Dict, model: str, u, stage: str, seed: int, K: int) -> str:
    rc = mods()["rc"]
    pin = cfg["pin_model_version"][model]
    blob = json.dumps({"pin": pin, "fid": FID_ALIASES[cfg["fidelity"]], "u": np.asarray(u, "<f8").tobytes().hex(),
                       "model": model, "stage": stage, "seed": int(seed), "K": int(K), "course": rc.VERSION,
                       "score": SCORING_VERSION, "w": cfg["weights"], "genes": PG_GENES_HASH()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def PG_GENES_HASH():
    from . import phase4_ga as PG
    return hashlib.sha256(json.dumps(PG.GENES).encode()).hexdigest()[:16]


_CFG = {}


def _winit(cfg):
    blas_single_thread_or_die()
    _CFG["cfg"] = cfg
    mods()


def _job(a):
    model, u, stage, seed = a
    r = fly_one(_CFG["cfg"], model, u, stage, seed)
    return {k: v for k, v in r.items() if k != "gates"} | {"gates": r["gates"]}


# ------------------------------------------------------------------------------------------------ course seeds
# seed_scheme "run_seed_v2" (opt-in, Corleone 04:59 PT, coordinated with Sim Bridge): train seed = sha256 of
# (run_seed, gen, k, aircraft) truncated to 64 bits; hold-out = sha256(run_seed, "holdout", j, aircraft). Uses Sim Bridge's
# ring_course.course_seed(run_seed, gen, k, aircraft, holdout=) when it exists (5-arg form); else this local formula,
# which a test compares with Sim Bridge's once it lands. Default scheme (s1) = ring_course.train_seeds / holdout_seeds.
def _local_seed(run_seed: int, gen, k: int, aircraft: str, holdout: bool = False) -> int:
    # = Sim Bridge ring_course 1.2 course_seed_string / course_seed (spec v0.6 s4): sha256 of the UTF-8 string, first 8 bytes big-endian
    g = "holdout" if holdout else str(int(gen))
    tag = f"p4course-seed/1|run={int(run_seed)}|gen={g}|k={int(k)}|ac={aircraft}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(tag).digest()[:8], "big")


def _sb_seed_fn():
    import inspect
    f = getattr(mods()["rc"], "course_seed", None)
    try:
        ps = list(inspect.signature(f).parameters)
    except (TypeError, ValueError):
        return None
    return f if "aircraft" in ps else None


def seed_source() -> str:
    return "sim_bridge.ring_course.course_seed" if _sb_seed_fn() else "evolution.phase4_loop._local_seed (SB helper not yet live)"


def train_seeds_v2(run_seed: int, model: str, gen: int, K: int) -> List[int]:
    f = _sb_seed_fn()
    return [int(f(run_seed, gen, k, model)) if f else _local_seed(run_seed, gen, k, model) for k in range(K)]


def holdout_seeds_v2(run_seed: int, model: str, H: int) -> List[int]:
    f = _sb_seed_fn()
    return [int(f(run_seed, None, j, model, holdout=True)) if f else _local_seed(run_seed, None, j, model, True) for j in range(H)]


def seeds_for(cfg: Dict, model: str, gen: int = None, holdout: bool = False) -> List[int]:
    rg, rc = cfg["rings"], mods()["rc"]
    if rg.get("seed_scheme", "legacy") == "run_seed_v2":
        rs = int(rg["run_seed"])
        return holdout_seeds_v2(rs, model, int(rg["holdout_courses"])) if holdout else \
            train_seeds_v2(rs, model, gen, int(rg["courses_per_genome"]))
    return rc.holdout_seeds(int(cfg["seed"]), int(rg["holdout_courses"])) if holdout else \
        rc.train_seeds(int(cfg["seed"]), gen, int(rg["courses_per_genome"]))


# ------------------------------------------------------------------------------------------------ config
REQUIRED = ("phase", "seed", "fidelity", "ga", "rings", "weights", "aircraft", "profiles", "pin_model_version")


def is_phase4(user: Dict) -> bool:
    return isinstance(user, dict) and user.get("phase") == "phase4_rings"


def resolve_config(user: Dict, name: str) -> Dict:
    from . import phase4_eval as PE
    if not is_phase4(user):
        raise ValueError("not a phase4_rings config")
    miss = [k for k in REQUIRED if k not in user]
    if miss:
        raise ValueError(f"phase4 config missing {miss}")
    if user["fidelity"] not in FID_ALIASES:
        raise ValueError(f"phase4 fidelity must be one of {sorted(FID_ALIASES)}")
    if sorted(user["weights"]) != sorted(PE.P4_TERM_KEYS):
        raise ValueError("weights must cover exactly P4_TERM_KEYS")
    for a in user["aircraft"]:
        if a["profile"] not in user["profiles"] or a["name"] not in user["pin_model_version"]:
            raise ValueError(f"aircraft {a['name']}: profile/pin missing")
    rg = user["rings"]
    for k in ("courses_per_genome", "holdout_courses", "aggregate", "curriculum"):
        if k not in rg:
            raise ValueError(f"rings.{k} missing")
    if rg["curriculum"].get("mode", "floor") not in ("floor", "pass_rate_gate"):
        raise ValueError("rings.curriculum.mode must be 'floor' (default) or 'pass_rate_gate'")
    lf = rg["curriculum"].get("late_fallback")
    if lf is not None and (not isinstance(lf, dict) or set(lf) - {"enabled", "rule"} or not isinstance(lf.get("enabled"), bool)):
        raise ValueError("rings.curriculum.late_fallback = {enabled: bool, rule: 'generation_thirds'}")
    if rg.get("seed_scheme", "legacy") not in ("legacy", "run_seed_v2"):
        raise ValueError("rings.seed_scheme must be 'legacy' (default) or 'run_seed_v2'")
    c = json.loads(json.dumps(user))
    if rg.get("seed_scheme") == "run_seed_v2" and c["rings"].get("run_seed") is None:
        c["rings"]["run_seed"] = int.from_bytes(os.urandom(8), "little")   # new run -> fresh run_seed (logged)
        c["rings"]["run_seed_source"] = "os.urandom(8)"
    elif rg.get("seed_scheme") == "run_seed_v2":
        c["rings"]["run_seed_source"] = c["rings"].get("run_seed_source", "config/override")
    c["run_id"] = user.get("run_id") or f"{name.replace('_', '-')}-s{user['seed']}"
    return c


def blas_threads():
    try:
        from threadpoolctl import threadpool_info
        return max([i.get("num_threads", 1) for i in threadpool_info()] or [1])
    except ImportError:
        return int(os.environ.get("OPENBLAS_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "0")) or 0)


def blas_single_thread_or_die():
    n = blas_threads()
    if n != 1:
        raise RuntimeError(f"Phase 4 needs single-threaded BLAS (got {n}); export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                           "MKL_NUM_THREADS=1 before starting Python")


def check_pins(cfg):
    pins = mods()["pins"]
    bad = {a["name"]: (cfg["pin_model_version"][a["name"]], pins[a["name"]]["active"]) for a in cfg["aircraft"]
           if cfg["pin_model_version"][a["name"]] != pins[a["name"]]["active"]}
    fp4 = mods()["fp4"]
    import coupled_sim as cs
    live = {a["name"]: fp4.model_version(P4_FID, a["name"], cs.ROOT, cs_mode="active") for a in cfg["aircraft"]}
    bad.update({k: (cfg["pin_model_version"][k], v) for k, v in live.items() if v != cfg["pin_model_version"][k]})
    if bad:
        raise RuntimeError(f"model_version pin mismatch {bad}")
    return live


# ------------------------------------------------------------------------------------------------ GA
def run(cfg: Dict, out_root: str = None, workers: int = None) -> Dict:
    from . import phase4_ga as PG, phase4_eval as PE, rings as RR
    m = mods()
    rc = m["rc"]
    blas_single_thread_or_die()
    live = check_pins(cfg)
    out_root = out_root or os.path.join(HERE, "runs")
    rd = os.path.join(out_root, cfg["run_id"])
    if os.path.exists(os.path.join(rd, "summary.json")):
        raise RuntimeError(f"{rd} already finished; pick a new run_id")
    os.makedirs(rd, exist_ok=True)
    json.dump({"resolved": cfg}, open(os.path.join(rd, "config.json"), "w"), indent=1)
    G, N = int(cfg["ga"]["generations"]), int(cfg["ga"]["pop_size"])
    K, H = int(cfg["rings"]["courses_per_genome"]), int(cfg["rings"]["holdout_courses"])
    ag = cfg["rings"]["aggregate"]
    cur = cfg["rings"]["curriculum"]
    gcfg = PG.config(N)
    workers = workers or int(cfg.get("workers", max(1, (os.cpu_count() or 2) - 1)))
    cache_p = os.path.join(rd, "cache.jsonl")
    cache = {}
    if os.path.exists(cache_p):
        for ln in open(cache_p):
            d = json.loads(ln)
            cache[d["key"]] = d["res"]
    cf = open(cache_p, "a")
    gl = open(os.path.join(rd, "genomes.jsonl"), "a")
    sl = open(os.path.join(rd, "seeds.jsonl"), "a")
    seed_src = seed_source()
    json.dump({"run_id": cfg["run_id"], "seed_scheme": cfg["rings"].get("seed_scheme", "legacy"),
               "run_seed": cfg["rings"].get("run_seed"), "run_seed_source": cfg["rings"].get("run_seed_source"),
               "seed_source": seed_src, "course_generator": rc.VERSION, "fd_dir": os.path.relpath(FD_DIR, TEAM),
               "model_versions": live, "curriculum": cfg["rings"]["curriculum"],
               "elite_rescore": "every member incl. elites is re-flown on the generation's fresh K courses"},
              open(os.path.join(rd, "run.json"), "w"), indent=1)
    t_wall0, cpu0 = time.time(), time.process_time()
    summary = {"run_id": cfg["run_id"], "run_seed": cfg["rings"].get("run_seed"), "seed_source": seed_src, "fd_dir": os.path.relpath(FD_DIR, TEAM), "model_versions": live,
               "course_generator": f"sim_bridge.ring_course {rc.VERSION}", "guidance": cfg.get("guidance", "genome"),
               "genes_hash": PG_GENES_HASH(), "aircraft": []}
    child_cpu = 0.0
    with ProcessPoolExecutor(max_workers=workers, initializer=_winit, initargs=(cfg,)) as ex:
        def evaluate(model, pop, stage, seeds):
            jobs, keys = [], []
            for u in pop:
                for s in seeds:
                    k = cache_key(cfg, model, u, stage, s, len(seeds))
                    keys.append(k)
                    if k not in cache:
                        jobs.append((k, (model, [float(x) for x in u], stage, int(s))))
            uniq = {}
            for k, a in jobs:
                uniq.setdefault(k, a)
            for k, res in zip(uniq, ex.map(_job, list(uniq.values()))):
                cache[k] = res
                cf.write(json.dumps({"key": k, "res": res}) + "\n")
            cf.flush()
            return [[cache[keys[i * len(seeds) + j]] for j in range(len(seeds))] for i in range(len(pop))]

        for ai, a in enumerate(cfg["aircraft"]):
            model = a["name"]
            rng = np.random.default_rng([int(cfg["seed"]), zlib.crc32(model.encode())])
            pop = PG.generation_zero(rng, N)
            hist, stages, gens_log = [], [], []
            t_ac = time.time()
            for g in range(G):
                if cur.get("mode", "floor") == "pass_rate_gate":     # opt-in (phase4 configs only)
                    stage = RR.curriculum_stage_gated(g, hist, cur["promote_pass_rate"], cur["promote_streak"])
                    if (cur.get("late_fallback") or {}).get("enabled"):   # optional by-generation floor (OFF in s2)
                        fl = RR.STAGE_ORDER[min(2, (3 * g) // max(G, 1))]
                        stage = max(stage, fl, key=RR.STAGE_ORDER.index)
                else:                                                # default: s1 behaviour (generation floor + promotion)
                    stage = RR.curriculum_stage(g, G, hist, cur["promote_pass_rate"], cur["promote_streak"])
                stages.append(stage)
                seeds = seeds_for(cfg, model, g)
                sl.write(json.dumps({"aircraft": model, "gen": g, "stage": stage, "train_seeds": seeds,
                                     "run_seed": cfg["rings"].get("run_seed"), "seed_source": seed_src}) + "\n"); sl.flush()
                per = evaluate(model, pop, stage, seeds)
                aggs = [PE.aggregate_courses(rs, ag["mode"], ag["alpha"], ag["blend"]) for rs in per]
                costs = np.array([x["cost"] for x in aggs])
                o = PG.rank_order(costs)
                for rank, i in enumerate(o):
                    gl.write(json.dumps({"aircraft": model, "gen": g, "rank": rank, "idx": int(i), "id": f"{model}:g{g}:r{rank}",
                                         "stage": stage, "seeds": seeds, "u": [float(x) for x in pop[i]], "cost": aggs[i]["cost"],
                                         "mean": aggs[i]["mean"], "cvar": aggs[i]["cvar"], "pass_rate": aggs[i]["pass_rate"],
                                         "hard_fails": aggs[i]["hard_fails"], "terms": aggs[i]["terms"],
                                         "courses": [{"seed": r["seed"], "cost": r["cost"], "hard_fail": r["hard_fail"],
                                                      **{k: r["summary"][k] for k in ("passes", "misses", "finished", "t_last", "mean_rho_m", "mean_miss_m")}}
                                                     for r in per[i]]}) + "\n")
                gl.flush()
                b = int(o[0])
                hist.append(aggs[b]["pass_rate"])
                nhard = sum(1 for rs in per for r in rs if r["hard_fail"])
                gens_log.append({"gen": g, "stage": stage, "seeds": seeds, "best": aggs[b]["cost"], "best_pass_rate": aggs[b]["pass_rate"],
                                 "median": float(np.median(costs)), "pop_pass_rate": float(np.mean([x["pass_rate"] for x in aggs])),
                                 "hard_fail_courses": nhard, "n_courses": N * K})
                print(f"[{model}] g{g} {stage} best {aggs[b]['cost']:.4f} pass {aggs[b]['pass_rate']:.3f} hard {nhard}", flush=True)
                ranked = pop[o]
                if g < G - 1:
                    pop = PG.next_generation(rng, ranked, gcfg)
            best_u = ranked[0]
            best_per = per[int(o[0])]
            best_agg = aggs[int(o[0])]
            hseeds = seeds_for(cfg, model, holdout=True)
            sl.write(json.dumps({"aircraft": model, "holdout_seeds": hseeds, "stage": stages[-1],
                                 "run_seed": cfg["rings"].get("run_seed"), "seed_source": seed_src}) + "\n"); sl.flush()
            hold = evaluate(model, [best_u], stages[-1], hseeds)[0]
            hagg = PE.aggregate_courses(hold, ag["mode"], ag["alpha"], ag["blend"])

            def sm(rs, key):
                v = [r["summary"][key] for r in rs if r["summary"].get(key) is not None]
                return (math.fsum(v) / len(v)) if v else None
            ent = {"aircraft": model, "best_id": f"{model}:g{G-1}:r0", "best_u": [float(x) for x in best_u],
                   "best_genes": PG.decode(best_u, model), "stages": stages, "generations": gens_log,
                   "train": {"cost": best_agg["cost"], "mean": best_agg["mean"], "pass_rate": best_agg["pass_rate"],
                             "terms": best_agg["terms"], "hard_fails": best_agg["hard_fails"], "seeds": seeds_for(cfg, model, G - 1),
                             "mean_rho_m": sm(best_per, "mean_rho_m"), "mean_miss_m": sm(best_per, "mean_miss_m"),
                             "t_last_mean": sm(best_per, "t_last"), "nominal_time_mean": sm(best_per, "nominal_time_s"),
                             "finished": sum(r["summary"]["finished"] for r in best_per)},
                   "holdout": {"cost": hagg["cost"], "mean": hagg["mean"], "pass_rate": hagg["pass_rate"], "terms": hagg["terms"],
                               "hard_fails": hagg["hard_fails"], "seeds": hseeds, "stage": stages[-1],
                               "mean_rho_m": sm(hold, "mean_rho_m"), "mean_miss_m": sm(hold, "mean_miss_m"),
                               "t_last_mean": sm(hold, "t_last"), "nominal_time_mean": sm(hold, "nominal_time_s"),
                               "finished": sum(r["summary"]["finished"] for r in hold),
                               "per_course": [{"seed": r["seed"], "cost": r["cost"], "passes": r["summary"]["passes"]} for r in hold]},
                   "holdout_minus_train_cost": hagg["cost"] - best_agg["cost"],
                   "wall_s": time.time() - t_ac}
            summary["aircraft"].append(ent)
            json.dump(summary, open(os.path.join(rd, "summary.partial.json"), "w"), indent=1)
    cf.close(); gl.close(); sl.close()
    import resource
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    summary["wall_s"] = time.time() - t_wall0
    summary["cpu_s"] = {"parent": time.process_time() - cpu0, "children": ru.ru_utime + ru.ru_stime}
    summary["workers"] = workers
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    return summary


# ------------------------------------------------------------------------------------------------ trajectory export
def _quat_enu(phi, th, psi):
    sys.path.insert(0, TEAM)
    from evolution.validate_traj import M_EN, euler_to_matrix_ned
    R = M_EN @ euler_to_matrix_ned(np.array([phi]), np.array([th]), np.array([psi]))[0]
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]])); j, k = (i + 1) % 3, (i + 2) % 3
        S = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / S; q[1 + i] = 0.25 * S; q[1 + j] = (m[j, i] + m[i, j]) / S; q[1 + k] = (m[k, i] + m[i, k]) / S
    if q[0] < 0:
        q = [-x for x in q]
    return q


def export_traj(cfg, model, u, stage, seed, run_id, gen, out_dir, label, M=None):
    """Re-fly one (genome, course) with recording; write ga-flightsim-traj/2 + course/gates blocks (Sim Bridge spec 5)."""
    rc = mods()["rc"]
    from sim_bridge import rings as sbr
    res = fly_one(cfg, model, u, stage, seed, record=True, M=M)
    r, course = res["_flight"], res["_course"]
    t = np.asarray(r["t"], float)
    pos = np.asarray(r["pos"], float)
    h0 = float(pos[0, 2])
    enu = np.stack([pos[:, 1], pos[:, 0], pos[:, 2] - h0], 1)
    dt = 1.0 / 30.0
    vel = np.gradient(enu, t, axis=0, edge_order=2)
    att = np.asarray(r["att"], float)
    dec = 4
    cmd = res["_cmd"]
    rows = []
    for i in range(len(t)):
        q = _quat_enu(*att[i])
        k = min(int(round(t[i] * 120)), len(cmd) - 1)
        th, el, ai, ru = cmd[k]
        psi = att[i, 2] % (2 * math.pi)
        rows.append([round(float(t[i]), 4), *[round(float(x), 3) for x in enu[i]], *[round(float(x), 8) for x in q],
                     *[round(float(x), 4) for x in vel[i]], round(float(pos[i, 2]), 3), round(float(att[i, 0]), 7),
                     round(float(att[i, 1]), 7), round(float(psi), 7), round(th, 6), round(el, 6), round(ai, 6), round(ru, 6),
                     round(float(r["v_kcas"][i]), 3), round(float(r["nz"][i]), 4)])
    chans = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "alt_msl_m", "phi", "theta", "psi",
             "throttle", "elevator", "aileron", "rudder", "kcas", "nz"]
    events = [{"t": 0.0, "type": "start", "detail": f"ring course {model} {stage} seed {seed}, {course['start']['alt_ft']:.0f} ft, {course['start']['kcas']:.0f} KCAS"}]
    for gt in res["gates"]:
        events.append({"t": float(gt["t"]), "type": "gate_pass" if gt["result"] == "pass" else "gate_miss",
                       "detail": f"ring {gt['k']} {gt['result']}" + (f" rho {gt['rho_m']:.1f} m" if gt.get("rho_m") is not None else "")})
    events.append({"t": float(t[-1]), "type": "end" if r["status"] == "ok" else "terminated", "detail": r["status"]})
    events.sort(key=lambda e: e["t"])
    gsum = {k: v for k, v in res["summary"].items() if k != "carried"}
    surf_ch = sbr.ctrl_surface_channels(r["surfaces_120hz"]) if "t_s" in (r.get("surfaces_120hz") or {}) else None
    doc = {"schema": TRAJ_SCHEMA, "run_id": run_id, "aircraft": model, "jsbsim_version": "fd-pin p4cs", "git_sha": "local-unpushed",
           "seed": int(cfg["seed"]), "generation": int(gen), "fitness": float(res["cost"]), "fitness_sense": "min",
           "fitness_doc": "Phase 4 per-course cost (er-p4-score/1) of this course; GA cost = blend over K courses",
           "scenario_index": 0, "scenario_id": f"{model}:{label}:{seed}", "scenario_cost": float(res["cost"]), "status": r["status"],
           "genome": {k: float(v) for k, v in res["_genes"].items()},
           "frame": {"origin_lat_deg": 0.0, "origin_lon_deg": 0.0, "origin_alt_m": h0, "axes": "ENU metres, x=east y=north z=up",
                     "attitude": "quat body->ENU [w,x,y,z]", "body_axes": "JSBSim body FRD: x forward, y right wing, z down",
                     "attitude_source_of_truth": "quaternion; phi/theta/psi are HUD-only (JSBSim Euler ZYX vs local NED)",
                     "origin_doc": "lat/lon = JSBSim default IC (profile has no origin); FD pos is N/E/Up, z = altitude; ENU x=E y=N z=Up-h0"},
           "units": {"phi": "rad", "theta": "rad", "psi": "rad", "controls": "norm -1..1, throttle 0..1"},
           "dt_s": dt, "sim_dt_s": 1.0 / 120.0, "sample_hz": 30,
           "target": {"alt_m": h0, "steps": [], "speed_kcas": float(course["start"]["kcas"]), "reference": "ring course (see course block)"},
           "wind": {"north_mps": 0.0, "east_mps": 0.0, "gust_sigma_mps": 0.0},
           "control_conventions": "JSBSim fcs/*-cmd-norm: elevator + = trailing-edge down (nose down); aileron + = right roll; rudder + = nose left; throttle 0..1",
           "controls_timing": "pre_step", "events": events,
           "course": sbr.course_block(course, h0), "gates": res["gates"], "gate_summary": gsum,
           "p4": {"model_version": res["model_version"], "stage": stage, "course_seed": int(seed), "label": label, "terms": res["terms"],
                  "hard_fail": res["hard_fail"], "carried": res["summary"].get("carried")},
           **({"ctrl_surfaces": surf_ch} if surf_ch else {}),
           "channels": chans, "data": rows}
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{model}_{run_id}_g{gen}_{label}_{seed}.json"
    json.dump(doc, open(os.path.join(out_dir, fn), "w"), separators=(",", ":"))
    return fn, doc, res


def main(argv=None, user=None, name=None):
    ap = argparse.ArgumentParser(description="Phase 4 ring-course GA")
    ap.add_argument("--config")
    ap.add_argument("--run-id")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--run-seed", type=int, help="replay: reuse a run's ring run_seed (run_seed_v2 scheme)")
    a, _ = ap.parse_known_args(argv)
    if user is None:
        user = json.load(open(a.config))
        name = os.path.splitext(os.path.basename(a.config))[0]
    if a.run_id:
        user["run_id"] = a.run_id
    if a.run_seed is not None:
        user.setdefault("rings", {})["run_seed"] = a.run_seed
        user["rings"]["run_seed_source"] = "--run-seed override"
    cfg = resolve_config(user, name)
    s = run(cfg, workers=a.workers)
    print(json.dumps({x["aircraft"]: [x["train"]["cost"], x["holdout"]["cost"]] for x in s["aircraft"]}))


if __name__ == "__main__":
    main()
