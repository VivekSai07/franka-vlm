# Phase A — Environment, Simulation, Task Generator

Study notes for the ReflectVLM simulation stack (`reflect-vlm-reference/roboworld/`),
written while reproducing the Phase A milestones on the 8GB laptop (WSL2).

## The task: board + pegs assembly

Every task is a procedurally generated "puzzle": a **board** (`brick_1`, a voxel slab
with holes cut into it) sits on a table, and 3–8 **pegs** (bricks, nails, or shaped
pegs — `brick_2..brick_9` at most, since the generator caps brick ids at 9) must
each be inserted into their matching hole. A Franka
Panda arm does the manipulation. The episode succeeds when every peg's `_align` site
coincides with its `{peg}_hole_align` site on the board (position error ≤ 5mm,
orientation error ≤ 0.02, checked by `env.is_success()`).

`generate_xml(seed)` (in `roboworld/envs/generator.py`) is fully deterministic per
seed: it seeds `np.random`/`random`, builds a voxel `Board`, carves bricks/nails/pegs
into it, and returns `(XmlMaker, info)` where `info` carries:

- `n_bodies` — board + number of pegs
- `brick_descriptions` — e.g. `{"brick_2": "red block", "brick_4": "yellow nail"}`;
  the first word is the **color label** the VLM uses to name objects in actions
- `dependencies` — a set of `(u, v)` brick-id pairs

## What `dependencies` encodes

When the generator places brick `v` so that its voxels intersect the voxels already
claimed by brick `u` (bricks stack/interlock on the board), it records `(u, v)`:
**`u` must be inserted before `v`** — otherwise `v` physically blocks `u`'s hole.
It is a DAG over the pegs; `AssemblyOracle` runs incremental topological ordering
over it (`in_deg` counting) to decide which pegs are `READY` vs `BLOCKED_P`
(predecessor missing) vs `BAD_B` (inserted out of order and now blocking others —
must be removed first). The randomized partial assembly at `reset()` can start the
board in a "some pegs already inserted, possibly wrongly" state, which is what makes
long-horizon recovery behavior necessary.

## The 5 action primitives

`ACTION_PRIMITIVES = ["pick up", "put down", "insert", "reorient", "done"]`, always
paired with a color label (`pick up red`), except `done`. `env.act_txt("pick up red")`:

1. maps color → body name via `env.peg_colors`
2. dispatches to `act_pick_up/act_put_down/act_insert/act_reorient`
3. each of those runs a **full scripted primitive** (the env's `control_mode` is
   hard-asserted to `"script"` — there is no low-level action interface in this
   release): waypoint `goto`s driven by a damped-least-squares differential-IK
   controller (`DiffIKNullspaceController`, adapted from mjctrl) with nullspace
   biasing toward the home pose, plus grasp-site selection, fixture-assisted
   reorientation, and weld-equality "magic attaching" for stable grasps
4. returns an **err code**: `0` OK; `pick up` → `-1` another object already in hand,
   `-2` target already in hand; `put down`/`insert`/`reorient` → `-1` target not in
   hand. Precondition failures return immediately without moving the arm.

So one "action" for the VLM = seconds of simulated robot motion. The policy's job is
purely the *symbolic* sequencing; the motor control is canned.

## Medium vs hard

`run.py`'s `build_env` filters generated boards by body count:
`n_bodies > 5` → **hard**, `n_bodies <= 5` → **medium**. The eval scripts all use
`--level=hard`. Boards that don't match the requested level are skipped by
incrementing the seed until one matches.

## Environment setup on this machine (WSL2, RTX PRO 2000 8GB)

- Reused the existing `reflect-vlm-reference/.venv` (Python 3.10.12) which already
  had the PLAN.md "fast path": `mujoco==3.1.2 gym==0.26.2 imageio imageio-ffmpeg
  moviepy numpy` (+ glfw, PyOpenGL).
- Added for `run.py`: `wandb ml_collections pandas`, then
  `pip install -e . --no-deps` so `roboworld` imports without pulling torch.
- **WSLg GLFW window works**: `interact.py` opens its MuJoCo viewer natively
  (DISPLAY=:0). Note: on the `exit` path the script returns without `env.close()`,
  so the passive-viewer thread keeps the process alive — close the window/Ctrl-C.
- **Offscreen rendering: the WSL2 D3D12 GPU driver is broken for sustained use.**
  The PLAN.md-flagged "verify EGL" risk is real, and worse than expected:
  - Single renders pass under `MUJOCO_GL=egl` (mesa d3d12 backend), but after
    ~10–250 renders the driver drops the device (`D3D12: Removing Device`) and
    the process dies — SIGSEGV inside `Renderer.render()` on one run, heap
    corruption (`mremap_chunk(): invalid pointer` → SIGABRT) on another.
  - `MUJOCO_GL=glfw` fails identically (~200 renders) — same d3d12 driver under
    both, so switching MuJoCo backends doesn't help.
  - No NVIDIA EGL ICD exists in this WSL2 install (only `50_mesa.json`), and
    libOSMesa isn't installed.
  - **Fix (zero-install): `LIBGL_ALWAYS_SOFTWARE=1`** forces mesa's llvmpipe
    software rasterizer — verified stable for 400 sustained renders. Rendering
    is sparse in these evals (~1 frame per decision step with `--record=False`),
    so the CPU rasterizer costs almost nothing; physics dominates.
  - Also run smoke evals with `--record=False`: recording only feeds the wandb
    video, while per-step PNGs, `goal.png`, and success rates are all produced
    regardless — and `--record=True` multiplies render count by ~100x.
  - Working incantation:
    `MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 python run.py ...`
  - (Kaggle note for Phase B: do NOT set `LIBGL_ALWAYS_SOFTWARE` there — its
    NVIDIA EGL stack should work natively; this is a WSL2-only workaround.)

## Verification results (2026-07-04)

- `scripts/interact.py`: GLFW window opens under WSLg ✔ (manual task completion
  done separately at the keyboard).
- Expert smoke run — `run.py --agent_type=expert --n_trajs=3 --level=hard
  --record=False` with `LIBGL_ALWAYS_SOFTWARE=1`: **Success rate 1.0 (3/3)**,
  28 decision steps total, per-step PNGs + `goal.png` + `meta.csv` written to
  `logs/eval_expert_smoke/` ✔ (matches the expected ~100% for the oracle).
- `mini_roboworld/explore_env.py` across seeds 1000000–1000004: printed
  n_bodies/level/labels/dependency-DAG per seed (8/7/6 bodies → hard,
  5 bodies → medium — consistent with the `n_bodies > 5` cutoff), and the
  act_txt demo returned exactly the predicted err codes
  (`insert` not-in-hand → -1, `pick up` → 0, re-`pick up` → -2, pick while
  hands full → -1, `insert` → 0 with `object_is_success` flipping to True) ✔.
- Neat observation from the demo seed: the oracle's *first* action was to pick
  up a `BAD (blocking other bricks)` peg — un-inserting a wrongly-placed piece
  before making forward progress. That recovery behavior is what the
  dependency DAG + `randomize_partial_assembly` combination is designed to
  force, and it's the behavior ReflectVLM's reflection mechanism is meant to
  teach the VLM.
