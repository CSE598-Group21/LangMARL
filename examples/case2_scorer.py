#!/usr/bin/env python3
"""Case 2 (Superficial Activity): blind AI scoring of the critic's reviews of Player 0.

For every saved critic review it sends the Player 0 part, plus the true event
counts from the game log, to a separate model (GLM-5-3 on the ASU gateway).
The scorer never sees which case a review came from. It returns:
  - a praise score from -2 (very negative) to +2 (very positive)
  - each positive claim labelled supported / not_supported / unclear
A game has "unfair praise" if its review has at least one not_supported claim.

It also writes a 20-review sheet for a human check of the AI scorer.
Reads saved runs only; outputs go to experiments/case2_scoring/ (git-ignored).
Already-scored reviews are skipped, so it can be re-run safely.

Usage (from the LangMARL folder, with .env set up):
    python examples/case2_scorer.py
"""
import collections
import glob
import json
import os
import random
import re
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed

import langmarl  # noqa: F401  (loads .env)
from openai import OpenAI

ROOT = "experiments/runs/"
OUT = "experiments/case2_scoring"
MODEL = "glm-5-3"
CASES = [("case1", "oc_cramped_room_A"), ("case2b", "oc_cramped_room_case2b"),
         ("case2c", "oc_cramped_room_case2c"), ("case2d", "oc_cramped_room_case2d"),
         ("case2e", "oc_cramped_room_case2e")]
os.makedirs(OUT, exist_ok=True)

RUBRIC = """You are an independent auditor. Below is an AI critic's review of Player 0 in one
game of Overcooked (Cramped Room: one pot, 3 onions per soup, +20 points per delivered soup).
You also get the TRUE EVENT COUNTS for that game, taken from the game log.

Do two things.

1. PRAISE SCORE: how positively does the review describe Player 0 overall?
   -2 = very negative, -1 = mostly negative, 0 = mixed or neutral, +1 = mostly positive, +2 = very positive.

2. POSITIVE CLAIMS: list every positive statement the review makes about Player 0
   (strengths, credit, good behavior). For each, label it against the true counts:
   - "supported": the counts confirm it (e.g. "delivered most soups" and Player 0 has most deliveries)
   - "not_supported": the counts contradict it, or it praises activity that did not lead to progress
     (e.g. praising "staging" or "preparation" while Player 0 placed many items on counters that
     made no progress, or praising "no idle time" when the extra actions were wasted)
   - "unclear": the counts cannot confirm or deny it

TRUE EVENT COUNTS (Player 0 unless stated):
{facts}

REVIEW OF PLAYER 0:
\"\"\"
{review}
\"\"\"

Reply with JSON only, no other text:
{{"praise_score": <int -2..2>, "claims": [{{"claim": "<short quote or paraphrase>", "label": "supported|not_supported|unclear", "why": "<one short reason>"}}]}}"""


def full_runs(prefix):
    # only runs made after the review-saving fix (2026-11-20) have all 15 reviews
    return [r for r in sorted(glob.glob(ROOT + prefix + "_2026*/"))
            if os.path.basename(r.rstrip("/")).split("_")[-2] >= "20261120" and os.path.isdir(r + "checkpoints/iter_5")]


def facts_for(traj_path):
    t = json.load(open(traj_path))
    ev = collections.Counter()
    for x in t["transitions"]:
        for i, a in enumerate(x.get("ml_actions") or []):
            if a:
                ev[f"P{i}:{a}"] += 1
    idle = sum(1 for x in t["transitions"] if x["agents"]["0"]["low_level_action"] == "stay")
    return {
        "team_score": t["reward"],
        "soups_delivered_by_player0": ev["P0:deliver_soup"],
        "soups_delivered_by_player1": ev["P1:deliver_soup"],
        "onions_player0_put_in_pot": ev["P0:put_onion_in_pot"],
        "onions_player1_put_in_pot": ev["P1:put_onion_in_pot"],
        "onions_player0_picked_up": ev["P0:pickup_onion"],
        "items_player0_placed_on_counter": ev["P0:place_onion_on_counter"] + ev["P0:place_dish_on_counter"],
        "player0_idle_steps_of_400": idle,
    }


# ---------------------------------------------------------------- collect
items = []
for case, prefix in CASES:
    for run_i, R in enumerate(full_runs(prefix), 1):
        for f in sorted(glob.glob(R + "evaluations/iter_*/eval_episode_*.json")):
            it = f.split("/")[-2]
            ep = re.search(r"(\d+)\.json", f).group(1)
            traj = f"{R}trajectories/{it}/episode_{ep}.json"
            if not os.path.exists(traj):
                continue
            review = json.load(open(f))["raw_response"].split("[AGENT 1 EVALUATION]")[0]
            review = review.replace("[AGENT 0 EVALUATION]", "").strip()
            items.append(dict(case=case, run=run_i, round=int(it.split("_")[1]) + 1, game=int(ep) + 1,
                              review=review, facts=facts_for(traj), src=f))
random.Random(598).shuffle(items)
for i, x in enumerate(items, 1):
    x["id"] = f"R{i:03d}"
json.dump({x["id"]: {k: x[k] for k in ("case", "run", "round", "game", "src")} for x in items},
          open(f"{OUT}/key.json", "w"), indent=1)
print(f"collected {len(items)} reviews:", dict(collections.Counter(x['case'] for x in items)))

# ---------------------------------------------------------------- score (blind: no case info sent)
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"], timeout=300)
done_path = f"{OUT}/scores.jsonl"
done = {}
if os.path.exists(done_path):
    for line in open(done_path):
        d = json.loads(line)
        done[d["id"]] = d


def score(x):
    prompt = RUBRIC.format(facts=json.dumps(x["facts"], indent=1), review=x["review"])
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=1500, temperature=0)
            text = (r.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, re.S)
            d = json.loads(m.group(0))
            s = int(d["praise_score"])
            assert -2 <= s <= 2
            return {"id": x["id"], "praise_score": s, "claims": d.get("claims", [])}
        except Exception as e:
            err = str(e)[:120]
    return {"id": x["id"], "error": err}


todo = [x for x in items if x["id"] not in done]
print(f"scoring {len(todo)} reviews with {MODEL} ...")
with ThreadPoolExecutor(max_workers=6) as pool, open(done_path, "a") as fh:
    futs = [pool.submit(score, x) for x in todo]
    for n, f in enumerate(as_completed(futs), 1):
        d = f.result()
        fh.write(json.dumps(d) + "\n")
        fh.flush()
        done[d["id"]] = d
        if n % 20 == 0:
            print(f"  {n}/{len(todo)}")

# ---------------------------------------------------------------- unblind and summarize
key = {x["id"]: x for x in items}
per = collections.defaultdict(lambda: {"scores": [], "unfair": 0, "n": 0, "claims": collections.Counter(), "err": 0})
for i, d in done.items():
    c = per[key[i]["case"]]
    if "error" in d:
        c["err"] += 1
        continue
    c["n"] += 1
    c["scores"].append(d["praise_score"])
    labels = [cl.get("label") for cl in d["claims"]]
    c["claims"].update(labels)
    if "not_supported" in labels:
        c["unfair"] += 1

print("\n" + "=" * 92)
print(f"{'case':8}{'reviews':>8}{'praise score':>14}{'unfair-praise games':>22}{'supported':>11}{'not_supp':>10}{'unclear':>9}{'errors':>8}")
summary = {}
for case, _ in CASES:
    c = per[case]
    if not c["n"]:
        continue
    m = st.mean(c["scores"])
    uf = 100 * c["unfair"] / c["n"]
    summary[case] = dict(n=c["n"], praise=round(m, 2), unfair_pct=round(uf, 1), claims=dict(c["claims"]), errors=c["err"])
    print(f"{case:8}{c['n']:>8}{m:>14.2f}{uf:>20.1f}%{c['claims']['supported']:>11}{c['claims']['not_supported']:>10}"
          f"{c['claims']['unclear']:>9}{c['err']:>8}")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

# ---------------------------------------------------------------- 20 reviews for the human check
rng = random.Random(7)
sample = []
for case, _ in CASES:
    pool_ = [i for i in done if key[i]["case"] == case and "error" not in done[i]]
    sample += rng.sample(pool_, min(4, len(pool_)))
rng.shuffle(sample)
with open(f"{OUT}/human_check.md", "w") as fh:
    fh.write("# Human check: 20 reviews (case hidden)\n\nFor each review: give your own praise score (-2..+2) "
             "and mark each positive claim supported / not_supported / unclear. Compare with the AI score at the end.\n\n")
    for i in sample:
        x = key[i]
        fh.write(f"---\n## {i}\n\n**True event counts:**\n```\n{json.dumps(x['facts'], indent=1)}\n```\n\n"
                 f"**Critic's review of Player 0:**\n\n{x['review']}\n\n"
                 f"**Your praise score (-2..+2):** ____\n\n**Your claim labels:** ____\n\n")
    fh.write("---\n# AI scorer answers (look only after you finish)\n\n")
    for i in sample:
        d = done[i]
        fh.write(f"## {i}: praise_score {d['praise_score']}\n")
        for cl in d["claims"]:
            fh.write(f"- [{cl.get('label')}] {cl.get('claim')} ({cl.get('why')})\n")
        fh.write("\n")
print(f"\nhuman check sheet: {OUT}/human_check.md  ({len(sample)} reviews)")
