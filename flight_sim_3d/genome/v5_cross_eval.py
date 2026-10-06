#!/usr/bin/env python3
"""Cross-evaluate v4 and v5 best genomes under both tasks (same scenario seeds) -> runs/v5_cross_eval.json."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, adapter
pairs=[("c172x",1,"hdg_after_c172x_s1","v5_sweep_w01"),("c172x",2,"hdg_after_c172x_s2","v5_c172x_s2"),("c172x",3,"hdg_after_c172x_s3","v5_c172x_s3"),("t38",1,"hdg_after_t38_s1","v5_t38_s1"),("b737",1,"hdg_after_b737_s1","v5_b737_s1")]
res=[]
for ac,s,r4,r5 in pairs:
    t4=adapter.load_task("phase1_v4",{"aircraft":ac}); t5=adapter.load_task("phase1_v5",{"aircraft":ac})
    g4=json.load(open(f"runs/{r4}/best_gains.json"))["gains"]; g5=json.load(open(f"runs/{r5}/best_gains.json"))["gains"]
    row={"ac":ac,"s":s}
    for tn,t in (("v4task",t4),("v5task",t5)):
        for gn,g in (("v4g",g4),("v5g",g5)):
            r=t.evaluate(g,t.make_scenarios(3,s)); row[f"{tn}_{gn}"]=r["cost"]; row[f"{tn}_{gn}_track"]=r["objectives"]["track_alt"]
    res.append(row)
    print(f"{ac} s{s}: v4 task: v4 genome {row['v4task_v4g']:.4f} (track {row['v4task_v4g_track']:.4f}) | v5 genome {row['v4task_v5g']:.4f} (track {row['v4task_v5g_track']:.4f})   v5 task: v4 genome {row['v5task_v4g']:.4f} | v5 genome {row['v5task_v5g']:.4f}")
json.dump(res,open("runs/v5_cross_eval.json","w"),indent=2)
