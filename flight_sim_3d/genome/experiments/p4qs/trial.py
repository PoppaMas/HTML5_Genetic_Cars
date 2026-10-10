"""Targeted trials: python trial.py MODEL STAGE SEEDIDX... with JSON variants on stdin-less argv: --var name:json"""
import sys, os, json, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import p4_guidance as PG, phase4_rings as R

def trial(model, stage, seeds, name, u=None, tune=None, over=None, variant="default"):
    P = R.load_preset(); u = R.identity_u(P["genes"]) if u is None else u
    out = []
    for sd in seeds:
        hook = (lambda g: {**g, **over}) if over else None
        s, r, c = PG.fly_one(u, P["genes"], model, stage, sd, variant=variant, tune=tune, gain_hook=hook)
        if s.get("cost") is None:
            out.append(dict(cost=None, st=s["status"])); continue
        T = s["terms"]; d = s.get("sb_diag", {})
        out.append(dict(cost=s["cost"], pr=s["pass_rate"], chat=T["J_chatter"], rate=T["J_rate_rms"], time=T["J_time"],
                        miss=d.get("mean_miss_m"), st=s["status"]))
    ok = [o for o in out if o["cost"] is not None]
    m = lambda k: float(np.mean([o[k] for o in ok if o.get(k) is not None] or [float("nan")]))
    res = dict(model=model, stage=stage, var=name, n=len(out), hard=len(out) - len(ok), pass_rate=m("pr"), cost=m("cost"),
               chatter=m("chat"), rate=m("rate"), J_time=m("time"), miss_m=m("miss"))
    print(json.dumps(res), flush=True)
    return res
if __name__ == "__main__":
    model, stage = sys.argv[1], sys.argv[2]
    RC = PG._sb(); seeds = [RC.course_seed(2, 0, k, model) for k in range(int(sys.argv[3]))]
    for v in sys.argv[4:]:
        name, js = v.split("=", 1); spec = json.loads(js)
        trial(model, stage, seeds, name, tune=spec.get("tune"), over=spec.get("over"))
