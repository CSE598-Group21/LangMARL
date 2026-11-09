#!/usr/bin/env python3
"""Scripted baseline: both Overcooked players are the Greedy planner (no LLM).

Plays N games per layout with the same horizon as the LangMARL runs and writes
the per-game rewards to experiments/baselines/greedy_greedy.json.

Usage:
    python examples/overcooked_greedy_baseline.py --layouts cramped_room forced_coordination --games 3
"""

import argparse
import json
import os
import statistics
import time

import langmarl
from langmarl.config.llm import LLMConfig
from langmarl.envs.overcooked.env import OvercookedEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", nargs="+", default=["cramped_room", "forced_coordination"])
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--out", default="experiments/baselines/greedy_greedy.json")
    args = parser.parse_args()

    # The env reads a model name at construction; Greedy players never call it.
    dummy = LLMConfig(name="unused", model_string="unused", api_key="unused")
    results = {}
    for layout in args.layouts:
        cfg = langmarl.OvercookedConfig(
            layout=layout, episode_horizon=args.horizon,
            p0_agent="Greedy", p1_agent="Greedy", llm=dummy,
        )
        env = OvercookedEnv(cfg)
        games = []
        for i in range(args.games):
            start = time.time()
            traj = env.collect_trajectory({}, {"task_id": f"episode_{i}"})
            games.append({
                "reward": traj.reward,
                "soups": int(traj.reward // 20),
                "steps": len(traj.steps),
                "seconds": round(time.time() - start, 1),
            })
        rewards = [g["reward"] for g in games]
        results[layout] = {"mean_reward": statistics.mean(rewards), "games": games}
        print(f"{layout}: mean={statistics.mean(rewards):.1f} rewards={rewards}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"horizon": args.horizon, "results": results}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
