"""Phase 4 opt-in family breeding (Corleone 10 Oct 2026). Used ONLY when config.breeding.mode == 'family';
the default 'isolated' never imports this module, so old behaviour (s1, s2, operator traces, identity hash) is untouched.

Every individual is a 29-gene vector u in [0,1] (phase4_ga.GENES), decoded per aircraft (phase4_ga.decode(u, model): per-aircraft
FD limits / gain scales), so a vector moved between T38 and f16 means 'the same fraction of that aircraft's range'.
Evaluation stays per aircraft (own stage, own courses, own pin, cache key carries the aircraft tag + breeding settings).

Mating: child of aircraft A = elite copies (cfg.elite, unchanged) + children. Parent 1 = flat-rank pick from A's own ranking.
With p = crossover_probability_within_family (and >1 family members) parent 2 = flat-rank pick from the ranking of a family
member drawn uniformly (may be A itself); otherwise parent 2 is from A (as isolated). Then ga.crossover_blocks + ga.mutate.
Block exchange: only blocks in breeding.exchange_blocks (default all three: guidance 12 | inner_loop 11 | mixing 6) may come from
parent 2; the others always come from parent 1 (own aircraft, e.g. ['guidance','inner_loop'] keeps each aircraft's own mixing block).
Members are never drawn from outside the family. Singleton families use phase4_ga.next_generation with the member's own rng.
Migration (optional): every `every_n_gens` generations, after breeding, member i of a family receives the top_k ranked genomes of
member i-1 (ring; for 2 members: a swap) in place of the LAST top_k slots of its next population (freshly bred, unevaluated
children = the 'worst' slots; elites are never replaced). Exactly top_k per member per event.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Tuple

import numpy as np

from . import ga

MODES = ("isolated", "family")
ALL_BLOCKS = ("guidance", "inner_loop", "mixing")      # Genome block order: guidance 12 | inner_loop 11 | mixing 6


def validate(b: Dict, names: List[str], pop: int, elite: int) -> None:
    if not isinstance(b, dict) or b.get("mode", "isolated") not in MODES:
        raise ValueError("breeding.mode must be 'isolated' (default) or 'family'")
    extra = set(b) - {"mode", "families", "crossover_probability_within_family", "migration", "exchange_blocks"}
    if extra:
        raise ValueError(f"unknown breeding keys {sorted(extra)}")
    if b.get("mode", "isolated") == "isolated":
        return
    fams = b.get("families")
    if not isinstance(fams, dict) or not fams:
        raise ValueError("breeding.families = {family: [aircraft, ...]} required for mode 'family'")
    seen = []
    for f, ms in fams.items():
        if not isinstance(ms, list) or not ms or not all(isinstance(x, str) for x in ms):
            raise ValueError(f"breeding.families[{f}] must be a non-empty list of aircraft names")
        seen += ms
    if len(seen) != len(set(seen)):
        raise ValueError("an aircraft may belong to only one family")
    xb = b.get("exchange_blocks")
    if xb is not None and (not isinstance(xb, list) or not xb or set(xb) - set(ALL_BLOCKS) or len(set(xb)) != len(xb)):
        raise ValueError("breeding.exchange_blocks must be a non-empty subset of ['guidance','inner_loop','mixing'] (default all three)")
    p = b.get("crossover_probability_within_family", 0.7)
    if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
        raise ValueError("crossover_probability_within_family must be in [0,1]")
    m = b.get("migration")
    if m is not None:
        if set(m) - {"every_n_gens", "top_k", "replace"} or m.get("replace", "worst") != "worst":
            raise ValueError("breeding.migration = {every_n_gens, top_k, replace: 'worst'}")
        if not (isinstance(m.get("every_n_gens"), int) and m["every_n_gens"] >= 1 and isinstance(m.get("top_k"), int)
                and 1 <= m["top_k"] <= pop - elite):
            raise ValueError("migration.every_n_gens >= 1 and 1 <= top_k <= pop_size - elite")


def families(b: Dict, names: List[str]) -> List[Tuple[str, List[str]]]:
    """Families restricted to the configured aircraft, config order; unlisted aircraft are singleton families."""
    out, used = [], set()
    for f, ms in b["families"].items():
        mem = [n for n in names if n in ms]
        if mem:
            out.append((f, mem)); used |= set(mem)
    out += [(n, [n]) for n in names if n not in used]
    return out


def family_rng(seed: int, run_seed, members: List[str]):
    return np.random.default_rng([int(seed), int(run_seed or 0), zlib.crc32(("family:" + "+".join(members)).encode())])


def next_generation(rng, ranked: Dict[str, np.ndarray], member: str, members: List[str], pcross: float, cfg, exchange=ALL_BLOCKS) -> Tuple[np.ndarray, List[Dict]]:
    own = ranked[member]
    n = own.shape[0]
    new = [own[i].copy() for i in range(min(cfg.elite, n))]
    meta = [{"origin": "elite", "parents": [{"aircraft": member, "rank": i}], "cross_family": False} for i in range(len(new))]
    while len(new) < cfg.pop_size:
        i = ga.flat_rank_select(rng, n, cfg.selection_p)
        pm, mem_b = member, own
        if len(members) > 1 and rng.random() < pcross:
            pm = members[int(rng.integers(len(members)))]
            mem_b = ranked[pm]
        nb = mem_b.shape[0]
        j = ga.flat_rank_select(rng, nb, cfg.selection_p)
        while pm == member and j == i:
            j = ga.flat_rank_select(rng, nb, cfg.selection_p)
        from . import phase4_ga as PG
        xblocks = [PG.BLOCKS[PG.BLOCK_ORDER.index(x)] for x in PG.BLOCK_ORDER if x in exchange]   # blocks not exchanged stay parent a's (own aircraft)
        child = ga.mutate(rng, ga.crossover_blocks(rng, own[i], mem_b[j], xblocks), cfg)
        new.append(child)
        meta.append({"origin": "child", "parents": [{"aircraft": member, "rank": int(i)}, {"aircraft": pm, "rank": int(j)}],
                     "cross_family": False, "cross_aircraft": pm != member})
    return np.array(new), meta


def migrate(new_pops: Dict[str, np.ndarray], metas: Dict[str, List[Dict]], ranked: Dict[str, np.ndarray],
            members: List[str], top_k: int) -> List[Dict]:
    """Ring migration; mutates new_pops/metas; returns the event list (one entry per moved genome)."""
    ev = []
    if len(members) < 2:
        return ev
    for idx, m in enumerate(members):
        donor = members[idx - 1]
        N = new_pops[m].shape[0]
        for r in range(top_k):
            slot = N - top_k + r
            new_pops[m][slot] = ranked[donor][r].copy()
            metas[m][slot] = {"origin": "migrant", "parents": [{"aircraft": donor, "rank": r}], "cross_family": False,
                              "cross_aircraft": True}
            ev.append({"to": m, "from": donor, "donor_rank": r, "slot": slot})
    return ev
