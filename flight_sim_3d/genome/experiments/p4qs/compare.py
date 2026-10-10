"""default vs qs, medium stage: identity + random gene vectors. usage: compare.py MODEL NSEEDS NRAND"""
import sys, os, json, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import p4_guidance as PG, phase4_rings as R
model, ns, nr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
P = R.load_preset(); RC = PG._sb(); seeds = [RC.course_seed(2, 0, k, model) for k in range(ns)]
us = {"identity": R.identity_u(P["genes"])}
rng = np.random.default_rng(11)
for i in range(nr): us[f"rand{i}"] = rng.random(len(P["genes"]))
res = []
for name, u in us.items():
    for var in ("default", "qs"):
        o = []
        for sd in seeds:
            s, r, c = PG.fly_one(u, P["genes"], model, "medium", sd, variant=var)
            if s.get("cost") is None: o.append(dict(hard=s["status"])); continue
            T = s["terms"]; o.append(dict(pr=s["pass_rate"], cost=s["cost"], chat=T["J_chatter"], miss=s["sb_diag"].get("mean_miss_m"), st=r["status"]))
        ok = [x for x in o if "pr" in x]
        row = dict(model=model, genes=name, var=var, hard=sum(1 for x in o if x.get("st", "ok") != "ok" or "hard" in x),
                   pass_rate=float(np.mean([x["pr"] for x in ok])) if ok else None,
                   cost=float(np.mean([x["cost"] for x in ok])) if ok else None,
                   chatter=float(np.mean([x["chat"] for x in ok])) if ok else None,
                   miss_m=float(np.mean([x["miss"] for x in ok if x["miss"] is not None] or [np.nan])) if ok else None)
        print(json.dumps(row), flush=True); res.append(row)
json.dump(res, open(os.path.join(os.path.dirname(__file__), f"compare_{model}.json"), "w"), indent=1)
