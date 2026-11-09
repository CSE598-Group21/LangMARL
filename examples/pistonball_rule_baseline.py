#!/usr/bin/env python3
"""Rule baseline for Pistonball (no LLM).

Each piston pushes up when it is right of the ball and retracts when it is
left of it, forming the left-low / right-high slope. Plays the same seeds,
horizon and reward as the LangMARL runs.

Usage:
    python examples/pistonball_rule_baseline.py --pistons 10 --games 3
"""

import argparse
import json
import os
import statistics

import numpy as np

from langmarl.envs.pistonball.observation import PistonballObservationFormatter, make_env

RETRACT, STAY, PUSH = 0, 1, 2


def play(num_pistons: int, max_cycles: int, seed: int) -> dict:
    env = make_env(num_pistons=num_pistons, max_cycles=max_cycles,
                    frame_size=(84, 84), stack_size=4, continuous=False, random_drop=True)
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    for _ in range(max_cycles):
        state = PistonballObservationFormatter.get_env_state(env)
        ball_x = state["ball_position"][0]
        actions = {}
        for name in env.possible_agents:
            if name not in obs:
                continue
            px = state["pistons"][env.possible_agents.index(name)]["position"][0]
            actions[name] = PUSH if px > ball_x else RETRACT
        obs, rewards, terms, truncs, _ = env.step(actions)
        total += float(np.mean(list(rewards.values()))) if rewards else 0.0
        steps += 1
        if any(terms.values()) or any(truncs.values()):
            break
    env.close()
    return {"seed": seed, "reward": round(total, 2), "steps": steps}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pistons", type=int, default=10)
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--max-cycles", type=int, default=900)
    p.add_argument("--out", default="experiments/baselines/pistonball_rule.json")
    a = p.parse_args()

    games = [play(a.pistons, a.max_cycles, seed) for seed in range(a.games)]
    mean = statistics.mean(g["reward"] for g in games)
    print(f"N={a.pistons}: mean={mean:.2f} games={games}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"pistons": a.pistons, "max_cycles": a.max_cycles,
                    "mean_reward": mean, "games": games}, f, indent=2)


if __name__ == "__main__":
    main()