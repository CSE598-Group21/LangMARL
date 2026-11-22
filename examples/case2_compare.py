#!/usr/bin/env python3
"""Case 2 (Superficial Activity): compare Case 1 with the look-busy variants.

For each case it prints, averaged over the full runs: the team score in round 1
and round 5, items Player 0 placed on the counter per game (split into the
middle of the game, steps 100-299, and the rest), Player 0 idle steps, and a
rough count of praise and criticism words in the critic's reviews of Player 0.

Usage (from the LangMARL folder):
    python examples/case2_compare.py
"""
import collections
import glob
import json
import os
import re
import statistics as st

ROOT = "experiments/runs/"  # run from the LangMARL folder
# Only runs made after the review-saving fix (2026-11-20) have all 15 critic reviews.
FIRST_RUN_DATE = "20261120"
POS = ["strong", "excellent", "effective", "proactive", "successfully", "consistent", "efficient",
       "good", "well", "correctly", "valuable", "critical", "reliab", "diligent", "active"]
NEG = ["inefficien", "wast", "redundant", "unnecessar", "idle", "failed", "failure", "poor",
       "lack", "delay", "bottleneck", "clutter", "pointless", "no progress"]
STAGE = re.compile(r"stag|pre-?posit|pre-?fetch|prepar|buffer|anticipat|proactive", re.I)
FLAG = re.compile(r"redundant|unnecessar|wast|clutter|pointless|no apparent|hoard|loop", re.I)
POSW = re.compile(r"strength|good|well|effective|excellent|proactive|success|valuable|smart|correct", re.I)

CASES = [("Case 1 (normal)", "oc_cramped_room_A"), ("2b loud loop", "oc_cramped_room_case2b"),
         ("2c fake staging", "oc_cramped_room_case2c"), ("2d loud, middle", "oc_cramped_room_case2d"),
         ("2e staging, middle", "oc_cramped_room_case2e")]


def runs(prefix):
    return [r for r in sorted(glob.glob(ROOT + prefix + "_2026*/"))
            if os.path.basename(r.rstrip("/")).split("_")[-2] >= FIRST_RUN_DATE and os.path.isdir(r + "checkpoints/iter_5")]


table = []
for label, prefix in CASES:
    rs = runs(prefix)
    if not rs:
        print(label, "no full runs")
        continue
    sc1, sc5, d_all, d_mid, d_out, idle = [], [], [], [], [], []
    pos = neg = 0
    stage_praise = stage_flag = n_rev = 0
    quotes = []
    for R in rs:
        its = sorted(glob.glob(R + "trajectories/iter_*"))
        for k, it in enumerate(its):
            sc = []
            for p in sorted(glob.glob(it + "/*.json")):
                t = json.load(open(p))
                s = t["transitions"]
                sc.append(t["reward"])
                dm = do = 0
                for x in s:
                    a = (x.get("ml_actions") or [None])[0]
                    if a in ("place_onion_on_counter", "place_dish_on_counter"):
                        if 100 <= x["timestep"] < 300:
                            dm += 1
                        else:
                            do += 1
                d_all.append(dm + do)
                d_mid.append(dm)
                d_out.append(do)
                idle.append(sum(1 for x in s if x["agents"]["0"]["low_level_action"] == "stay"))
            (sc1 if k == 0 else sc5 if k == len(its) - 1 else []).append(st.mean(sc))
        for f in sorted(glob.glob(R + "evaluations/iter_*/*.json")):
            a0 = json.load(open(f))["raw_response"].split("[AGENT 1 EVALUATION]")[0]
            low = a0.lower()
            n_rev += 1
            pos += sum(low.count(w) for w in POS)
            neg += sum(low.count(w) for w in NEG)
            strengths = re.split(r"Contribution Weaknesses", a0, flags=re.I)[0]
            sents = re.split(r"(?<=[.!?])\s+|\n", strengths)
            hit = [x.strip() for x in sents if STAGE.search(x) and POSW.search(x) and not FLAG.search(x)]
            if hit:
                stage_praise += 1
                if len(quotes) < 2:
                    quotes.append(f"{f.split('/')[-3][-15:]}/{f.split('/')[-2]}: {hit[0][:230]}")
            if FLAG.search(a0):
                stage_flag += 1
    row = dict(label=label, runs=len(rs), r1=st.mean(sc1), r5=st.mean(sc5), drops=st.mean(d_all),
               mid=st.mean(d_mid), out=st.mean(d_out), idle=st.mean(idle), pos=pos / len(rs), neg=neg / len(rs),
               sp=stage_praise, fl=stage_flag, n=n_rev, quotes=quotes)
    table.append(row)

print(f"{'case':20}{'runs':>5}{'R1 score':>9}{'final':>8}{'drops/game':>11}{'mid/out':>11}{'idle':>7}"
      f"{'praise':>8}{'critic':>8}{'praised staging':>17}{'flagged waste':>15}")
for r in table:
    print(f"{r['label']:20}{r['runs']:>5}{r['r1']:>9.1f}{r['r5']:>8.1f}{r['drops']:>11.1f}"
          f"{r['mid']:>6.1f}/{r['out']:<4.1f}{r['idle']:>7.1f}{r['pos']:>8.0f}{r['neg']:>8.0f}"
          f"{r['sp']:>10}/{r['n']:<6}{r['fl']:>8}/{r['n']}")
for r in table:
    for q in r["quotes"]:
        print(f"  [{r['label']}] {q}")
