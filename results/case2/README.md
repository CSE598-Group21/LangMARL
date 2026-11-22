# Case 2 (Superficial Activity) results

Cramped Room, Player 0 = learning AI (Gemma 4 31B), Player 1 = Greedy partner,
critic and optimizer = Qwen 3 27B, 5 rounds x 3 games x 400 steps.

| Case | Config | Look-busy note |
|---|---|---|
| Case 1 (normal) | `configs/overcooked/asu_cramped_room_A.json` | none |
| 2a weak note | `configs/overcooked/asu_cramped_room_case2.json` | mild note (2 runs) |
| 2b loud loop | `configs/overcooked/asu_cramped_room_case2b.json` | pick up / put down loop |
| 2c fake staging | `configs/overcooked/asu_cramped_room_case2c.json` | busy work framed as staging |
| 2d loud, middle | `configs/overcooked/asu_cramped_room_case2d.json` | loud loop, steps 100-299 only |
| 2e staging, middle | `configs/overcooked/asu_cramped_room_case2e.json` | staging, steps 100-299 only |

## Folders
- `logs/`: console output of each training run (`caseX_runN.log`).
- `scoring/`: blind AI scoring of the critic's reviews of Player 0 (GLM-5-3).
  - `summary.json`: praise score and share of games with unfair praise per case.
  - `scores.jsonl`: the scorer's answer for every review (by review id).
  - `key.json`: maps each review id back to its case, run, round and game.

## Reproduce
```bash
langmarl train --config configs/overcooked/asu_cramped_room_case2c.json   # one run
python examples/case2_compare.py   # behaviour and word-count summary
python examples/case2_scorer.py    # blind AI scoring (needs .env with the ASU key)
```
