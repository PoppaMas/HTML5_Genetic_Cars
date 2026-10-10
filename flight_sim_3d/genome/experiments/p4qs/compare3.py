"""default vs qs vs p4r1 on medium: usage compare3.py MODEL NSEEDS NRAND combo...   combo = variant@plant (e.g. default@cs qs@cs p4r1@cs1)"""
import sys, os, json, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import p4_guidance as PG, phase4_rings as R
model, ns, nr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); combos = sys.argv[4:]
P = R.load_preset(); RC = PG._sb(); seeds = [RC.course_seed(2, 0, k, model) for k in range(ns)]
best = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "runs", "p4_seed_c172x_best_s2.json")))["u"]
us = {"identity": R.identity_u(P["genes"]), "c172x_best": np.array(best)}
rng = np.random.default_rng(11)
for i in range(nr): us[f"rand{i}"] = rng.random(len(P["genes"]))
res = []
for name, u in us.items():
    for cb in combos:
        var, plant = cb.split("@"); caps = "clip" if var == "p4r1clip" else "range"; var = "p4r1" if var.startswith("p4r1") else var; o = []
        for sd in seeds:
            s, r, c = PG.fly_one(u, P["genes"], model, "medium", sd, variant=var, plant=plant, caps=caps)
            if s.get("cost") is None: o.append(dict(hard=1)); continue
            T = s["terms"]; o.append(dict(pr=s["pass_rate"], cost=s["cost"], chat=T["J_chatter"], hard=int(r["status"] != "ok")))
        ok = [x for x in o if "pr" in x]
        row = dict(model=model, genes=name, combo=cb, n=len(o), hard=sum(x["hard"] for x in o),
                   pass_rate=float(np.mean([x["pr"] for x in ok])) if ok else None,
                   chatter=float(np.mean([x["chat"] for x in ok])) if ok else None)
        print(json.dumps(row), flush=True); res.append(row)
json.dump(res, open(os.path.join(os.path.dirname(__file__), f"compare3_{model}_{combos[0].replace('@','_')}.json"), "w"), indent=1)
