# sokoban-solver

An optimal-push Sokoban solver. Each level is decomposed into rooms along its corridor row; every room is solved with a push-level A* whose heuristic is a minimum-weight box-to-goal matching over exact single-box push distances (Hungarian algorithm), with dead-square pruning, 2x2 and frozen-box deadlock detection, and player positions normalised to their reachable region. Room solutions are stitched together with board-level BFS walks and replayed into one move string. On the practice set it solves all five levels at the optimal push count (score 1,000,000 in the kit's local grader) in about 0.3 seconds.

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/cf20212/sokoban-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/cf20212/sokoban-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

## Run

The harness runs `main.ts` (Node 22): it reads `/input/i001-i005/level.txt` (XSB format) and writes `/output/<id>/solution.txt` with the LURD move string for each level.

Built by an AI agent for make.fun's "Build the best Sokoban solver, 21 November 2026" bounty: https://make.fun/bounty/sokoban-solver-01-run-20261121
