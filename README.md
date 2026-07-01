# franka-vlm workspace

Personal study/build workspace for learning to control a Franka arm with a VLM,
following [ReflectVLM](https://github.com/yunhaif/reflect-vlm) (arXiv 2502.16707)
as the primary reference.

The reference repo is *not* vendored here — it's cloned separately and read-only:
```
git clone https://github.com/yunhaif/reflect-vlm.git reflect-vlm-reference
```
(kept as a sibling directory, not inside this repo)

## Layout

- `notes/` — per-phase study notes (env/sim, VLM agent, reflection loop, ...)
- `mini_roboworld/` — own reimplementations of key pieces (action proposal + scoring,
  reflection heuristics, a simplified task generator) — not copies of the reference repo
- `experiments/` — eval logs/artifacts pulled back from Kaggle runs
- `requirements/` — environment definitions per compute target (WSL2 sim-only vs. Kaggle full)

See the phased plan for the full curriculum (env setup → base VLM → reflection loop →
reimplementation exercises → stretch goals).
