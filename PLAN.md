# Note for the executing agent (Claude Code, other laptop)

This file is the phased implementation plan for this project, written and approved
on a different machine. Pick it up from here and execute phase by phase.

**Suggested skills to invoke while working through this** (if available in your install):
- `superpowers:executing-plans` — this is exactly a written implementation plan meant
  to be executed across multiple sessions with review checkpoints at each phase boundary.
  Invoke it to drive execution phase-by-phase rather than running everything in one shot.
- `environment-awareness` — invoke at the start of a session before running setup
  commands. This repo's stack (flash-attn, bitsandbytes, deepspeed, `MUJOCO_GL=egl`) is
  Linux-only, so confirm WSL2/distro/GPU state before assuming a command will work.
- `superpowers:systematic-debugging` — use this instead of guessing when something fails
  (WSLg not rendering the interact.py window, EGL rendering failing on Kaggle, CUDA OOM
  when loading VLM+diffusion together, etc.) — several of these are flagged below as
  "verify, don't assume."
- `superpowers:verification-before-completion` (or `completion-gate`) — every phase below
  has an explicit Verification line; use this before marking a phase's milestone done.
- `graphify` — run `/graphify franka_vlm/workspace` after each phase closes out (starting
  after Phase A) to keep the knowledge graph of your own notes/code current. See the
  "Where /graphify Fits" section near the end of this plan.
- `superpowers:test-driven-development` — useful specifically for the Phase B/D
  reimplementation exercises where you're comparing your own script's output against a
  reference implementation's output (e.g. `propose_and_score.py` vs `LlavaAgent.act()`).

---

# ReflectVLM Study Plan: Learning to Control Franka with a VLM

## Context

You want to replicate and deeply understand [reflect-vlm](https://github.com/yunhaif/reflect-vlm) (arXiv 2502.16707, "Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation") to learn how VLMs can be used to control a Franka robot arm. `franka_vlm/` is currently empty, so this is a greenfield setup.

I cloned the actual repo (into a scratch dir, not the project) and read the source directly — not just the README — so the facts below (file paths, function signatures, exact prompt text, flag names) are verified against real code, not inferred. Where something couldn't be confirmed, it's explicitly flagged as "verify during Phase X."

**What the repo actually is:** A fine-tuned LLaVA-v1.5-13B VLM proposes text actions (5 primitives: `pick up`, `put down`, `insert`, `reorient`, `done` + an object color label) for a simulated Franka Panda arm assembling pegs into a board, in MuJoCo. Before executing, it "reflects": it imagines the likely next state either by (a) actually stepping the real MuJoCo sim forward and restoring state afterward, or (b) using a diffusion model (InstructPix2Pix-based) to generate an imagined next image — then re-prompts the VLM with that imagined image to confirm or revise the action before really executing it. **This is simulation-only** — no real Franka hardware integration exists in the repo (no frankapy/polymetis/franka_ros), only a MuJoCo MJCF model of the arm. Policy training code is *not* released (README: "coming soon") — only generic LLaVA fine-tuning scaffolding (`llava/train/`) exists, not the actual ReflectVLM post-training recipe.

**Compute reality (verified this session):**
- This machine's GPU: **NVIDIA GTX 1650, 4GB VRAM** (via `nvidia-smi`). Too tight for anything beyond authoring notes/code and pushing to GitHub — **this machine is not the dev environment for this project.**
- **Primary dev machine: your other laptop (RTX 2000-class, 8GB VRAM)**, via WSL2 there. 8GB is enough to comfortably debug the 13B VLM alone (4-bit quantized, ~7-9GB), but still not enough to hold VLM + diffusion model simultaneously, and too tight for real fine-tuning.
- **Phase B/C's actual model inference (and any fine-tuning) runs on Kaggle** (free T4x2/P100, 16GB, ~9-12h session cap, weekly quota) regardless of which laptop you're on — this is a hard requirement, not a comfort choice.
- Whichever machine runs WSL2, watch free disk space on the Windows-visible drive — don't cache multi-GB HF checkpoints or pip environments there; use WSL2's native ext4 filesystem instead.

## Workflow: Cross-Machine Sync via GitHub

You'll author the initial scaffold here, push to your own GitHub repo, then do all real work (WSL2 + Kaggle) on the 8GB laptop.

**On this machine** (one-time):
```powershell
cd "C:\Users\Vivek Sai\Downloads\franka_vlm"
mkdir workspace; cd workspace; git init
# create the initial structure (README, notes/, mini_roboworld/, experiments/, requirements/), commit
gh repo create <your-repo-name> --private --source=. --remote=origin   # or create on github.com and `git remote add origin <url>`
git push -u origin main
```
Do **not** push `reflect-vlm-reference/` to your own repo — it's just a clone of a public upstream repo; re-clone it fresh wherever you need it (8GB laptop, Kaggle) instead of forking it into your own remote.

**On the 8GB laptop:**
```bash
mkdir -p ~/dev/franka_vlm && cd ~/dev/franka_vlm
git clone <your-github-repo-url> workspace
git clone https://github.com/yunhaif/reflect-vlm.git reflect-vlm-reference
```

## Directory Structure (on the 8GB laptop, inside WSL2)

```
~/dev/franka_vlm/
├── reflect-vlm-reference/      # fresh clone of the upstream repo, read-only reference
└── workspace/                  # cloned from your GitHub repo — this is what /graphify points at
```

```
workspace/
├── README.md
├── notes/                  # phase-a-env-and-sim.md, phase-b-vlm-agent.md, phase-c-reflection-loop.md, ...
├── mini_roboworld/         # your own reimplementations (Phase D)
├── experiments/            # eval logs/artifacts pulled back from Kaggle
└── requirements/           # wsl2-cpu-mujoco.txt, kaggle-full.txt
```

**Raise WSL2 memory on the 8GB laptop** before any real work — create `.wslconfig` in that machine's Windows user profile:
```ini
[wsl2]
memory=10GB
processors=6
swap=8GB
```
Then `wsl --shutdown` and restart the distro; verify with `free -h`. (This machine's own WSL2 — Ubuntu-22.04, GPU passthrough already confirmed working, but no `.wslconfig` and only ~3.7GB allocated by default — doesn't need this since it's not the dev target, but the same fix applies if you ever do want to use it for lightweight sim-only work.)

## Phased Curriculum

Each phase: goal, files to study (verified paths), reproduce milestone, dissect/reimplement milestone, verification.

### Phase A — WSL2 env, MuJoCo sim, task generator, action primitives (no GPU-heavy deps needed)

Setup (on the 8GB laptop, inside WSL2):
```bash
cd ~/dev/franka_vlm/reflect-vlm-reference
python3 -m venv .venv && source .venv/bin/activate
# Fast path first — sim only, skip torch/transformers/diffusers for now:
pip install mujoco==3.1.2 gym==0.26.2 imageio==2.37.0 imageio-ffmpeg moviepy numpy
```

Study, in order:
1. `roboworld/constants.py` — confirmed `ACTION_PRIMITIVES = ["pick up", "put down", "insert", "reorient", "done"]`.
2. `scripts/interact.py` — interactive human-play script (`generate_xml(seed)` → `FrankaAssemblyEnv` → `AssemblyOracle`/`OracleAgent`). `render_mode='window'` is hardcoded, needs WSLg for the GLFW window — **verify WSLg renders it in Phase A**; if it fails, fall back to `MUJOCO_GL=egl` + `env.read_pixels()` saved as PNGs.
3. `roboworld/envs/generator.py` (855 lines) — procedural task generator. Confirmed: `COLORS` (9), `SHAPES` (15), voxel-based construction, seeded via `generate_xml(seed=...)`. "hard" vs "medium" level = `n_bodies > 5` vs `<= 5` (confirmed in `run.py`'s `build_env()`).
4. `roboworld/envs/mujoco/franka/franka_assembly.py` + `franka_env.py` — `env.act_txt(action_str)`, `env.read_pixels()`, `env.is_success()`, `env.goal_images`.
5. `roboworld/envs/mujoco/franka/control.py` — low-level control (not yet read line-by-line — **verify during Phase A** how a text primitive becomes joint motion, and confirm the `control_mode="script"` flag by grepping the two env files).
6. `roboworld/agent/oracle.py` — the scripted expert (`AssemblyOracle`/`OracleAgent`).

Reproduce:
- `python scripts/interact.py` — manually complete one task.
- Smoke-test the real `run.py` pipeline with zero GPU/VLM needed: `python run.py --agent_type=expert --n_trajs=3 --level=hard --record=True --save_dir=logs/eval_expert_smoke --logging.online=False` (use `--logging.online=False` to skip wandb login for now). Confirm per-trajectory PNGs, `goal.png`, and printed `Success: True/False` + success rate (should be near-100% since it's the ground-truth oracle).

Dissect:
- `workspace/mini_roboworld/`: standalone script that calls `generate_xml(seed=N)` directly, drives it with manual `act_txt` calls (inspect the returned `err` code), and prints `env_info` (peg_ids/labels/dependencies) across 5 seeds.
- `workspace/notes/phase-a-env-and-sim.md`: explain the board+pegs task, the 5 primitives, what `dependencies` encodes (insertion order constraints), and the medium/hard cutoff.

Verification: `interact.py` completes a task; expert smoke run produces PNGs + near-100% success rate.

### Phase B — Base VLM agent (`act()` deep-dive; inference on Kaggle)

Files (all read directly, confirmed):
- `roboworld/agent/llava.py`, class `LlavaAgent`:
  - `__init__(model_path, model_base=None, load_8bit=False, load_4bit=False, temperature=0.2, max_new_tokens=1024, conv_mode=None, debug=False)` — auto-picks `conv_mode="llava_v1"` for this checkpoint.
  - `act(image, goal_image, inp, next_image=None, num_propose_actions=1, return_score=False, temperature=0)` — builds image tensor via `process_images([goal_image, image] if next_image is None else [goal_image, image, next_image], ...)` (order: goal, current, [imagined future]). Calls `model.generate(..., num_return_sequences=num_propose_actions, output_scores=True, return_dict_in_generate=True)`, ranks candidates via `compute_transition_scores(..., normalize_logits=True)` summed to EOS.
- `roboworld/agent/utils.py` — `parse_act_txt` and `get_prompt`. Exact "propose" prompt (verbatim from source, typo included):
  > "There is a puzzle consisting of a board and several pieces with different colors on the table. The goal is to assemble the puzzle with the robot arm. In each step, one of the following four actions can be taken: pick up [obj], put down [obj], reorient [obj], and insert [obj], where [obj] refers to the piece to be manipulataed. The image of the goal state is: \<image\>. The image of the current state is: \<image\>. The most recently executed actions are: {history}. What action should be taken next? Note that [obj] should be a color chosen from the following list: {obj_labels}."
  ("done" is detected from raw output, not listed as a selectable primitive in-prompt.) The "reflect" variant adds `initial_plan` and a third `<image>` token for the imagined future state.

VRAM reality: 13B (even 4-bit) needs ~8-10GB+ comfortably — **run this phase's actual inference on Kaggle**, not locally.

Reproduce (Kaggle, GPU + internet on):
```bash
git clone https://github.com/yunhaif/reflect-vlm.git && cd reflect-vlm
pip install -e .
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # verify EGL works in Kaggle's container; osmesa is the CPU fallback
bash scripts/eval_base_vlm.sh   # override --n_trajs down to 5 first before a full 100-traj run
```
Confirmed flags: `--agent_type="llava"`, `--level='hard'`, `--oracle_prob=0`, `--model_path='yunhaif/ReflectVLM-llava-v1.5-13b-base'`, `--record=True`. Use `--logging.online=False` unless you've run `wandb login`.

Dissect:
- `workspace/mini_roboworld/propose_and_score.py` (Kaggle): load the model directly via `llava.model.builder.load_pretrained_model`, build the exact prompt yourself (don't import `get_prompt`), call `generate(..., num_return_sequences=4, output_scores=True, ...)` yourself, reimplement the transition-score ranking loop yourself. Compare your top-ranked action against `LlavaAgent.act(..., num_propose_actions=4, return_score=True)` on identical input — should match at temperature=0.
- `workspace/notes/phase-b-vlm-agent.md`: multi-image ordering, why log-prob-sum-to-EOS scoring.

Verification: Kaggle run produces `Success:` lines + `accumulated_success_rate` over ≥5 trajectories; your own ranking script matches `LlavaAgent.act()`'s output.

### Phase C — Reflection loop: sim-based and diffusion-based dynamics

Files (read in full):
- `run.py` — `imagine_with_sim(env, agent, first_action, goal_img, history, obj_labels, traj_dir, t)`: snapshots env state (`env.__getstate__()`), actually steps the real sim forward up to `imagine_future_steps` times (or until success/`"done"`), saves `sim-{t}-{plan}.png` frames, **restores state afterward** (`env.__setstate__`) — pure lookahead, no leakage into the real trajectory.
- `run.py` — `imagine_with_diffusion(...)`: same loop shape, but calls `diffusion_sim.forward(curr_image, act_text)` instead of stepping the sim, chaining its own outputs for multi-step lookahead (errors can compound), saves `gen-{t}-{plan}.png`. No state save/restore needed (real sim untouched).
- `run.py` `main()` per-step sequence: (1) propose via `agent.act(img, goal_img, inp)`, (2) if `--revise_action`, imagine via sim or diffusion (branch chosen by whether `--diffuser_pretrained_model` was passed), (3) build a second "reflect" prompt with the imagined image as a 3rd image and re-call `agent.act(...)`, (4) validate the (possibly revised) action format, falling back to the original if parsing fails, (5) execute for real via `env.act_txt()`.
- `roboworld/agent/diffuser.py`, `DiffusionSim`: wraps `StableDiffusionInstructPix2PixPipeline` (fp16), optional `unet_dir`/`vae_dir` overrides (EMA weights), fixed hyperparameters (`num_inference_steps=50`, `image_guidance_scale=1.5`, `guidance_scale=10`), **fixed seed** (`manual_seed(0)`) — deterministic per input.
- `scripts/diffusion_demo.py` — minimal standalone diffusion-only smoke test (no MuJoCo/LlavaAgent needed) — good first thing to run.

**Not confirmed — verify during Phase C:** no `load_4bit`/`load_8bit` flag appears wired into `run.py`'s `FLAGS_DEF` or either eval shell script, even though `LlavaAgent.__init__` supports them. If Kaggle VRAM is tight running VLM+diffusion together, you may need to patch `run.py`'s `LlavaAgent(...)` call yourself to add `load_4bit=True`, or load sequentially (VLM → propose+step → unload → diffusion → imagine → unload) instead of simultaneously.

Reproduce (Kaggle, sequential):
1. `python scripts/diffusion_demo.py` — confirm plausible edited output images.
2. `bash scripts/eval_reflect_vlm.sh sim` (small `--n_trajs` first) — confirmed: `model_path="yunhaif/ReflectVLM-llava-v1.5-13b-post-trained"`, `--revise_action=True`, `--imagine_future_steps=5`, `--level='hard'`, `--oracle_prob=0`.
3. `bash scripts/eval_reflect_vlm.sh diffusion` — adds `--diffuser_pretrained_model=yunhaif/ReflectVLM-diffusion`, routes to `imagine_with_diffusion`. Heaviest VRAM combination — watch for OOM.

Dissect:
- `workspace/mini_roboworld/toy_reflect_heuristic.py`: reimplement the env-snapshot/restore pattern using the oracle (not the VLM) for lookahead — deliberately skip the restore once to observe trajectory corruption, then fix it.
- A short script chaining 5 self-conditioned diffusion steps (like `imagine_with_diffusion`) to visually inspect compounding drift.
- `workspace/notes/phase-c-reflection-loop.md`: propose-then-reflect two-call pattern; why sim-based reflection needs state restore but diffusion doesn't; diffusion's fixed-seed determinism.

Verification: `diffusion_demo.py` output looks plausible; both `eval_reflect_vlm.sh` variants complete trajectories with logged success rates; compare against Phase B's base-VLM success rate on the same small n (the paper's core claim is that reflection improves this).

### Phase D — Reimplementation exercises (deepen understanding; mostly WSL2-local)

Extend `workspace/mini_roboworld/`:
1. Generalize Phase B's `propose_and_score.py` into a reusable module with your own prompt construction and scoring math (reference the originals for structure, don't copy-paste bodies).
2. A dependency-constraint-checking heuristic (using `env_info["dependencies"]` from Phase A) that vetoes/revises a proposed action without any imagined image at all — log explicitly in notes that this is a scaffolding exercise, not a real learned reflection mechanism.
3. `tiny_generator.py` — a simplified task generator (2-3 fixed pegs, simple cubes, no voxel geometry) that maps a seed to `peg_labels`/`dependencies` without the full 855-line machinery.
4. Run 1-3 locally under WSL2 (lightweight); item 1 needs *some* model — consider a small open VLM (ties into Phase E) or verify logic against dummy logits if Kaggle quota is precious.

Verification: each module runs standalone with sensible printed output; `workspace/notes/` gets a "what I now understand" reflection per module.

### Phase E (optional/stretch, in rough value-per-effort order)

1. Side-by-side comparison: real-sim lookahead vs. diffusion-imagined lookahead vs. actual stepped outcome, from the same starting state/action.
2. Swap a small open VLM into your Phase D harness to see how proposal quality degrades — the only Phase E item that might fit the local 4GB GPU if a genuinely small (sub-2B) model is chosen.
3. Full `n_trajs=100` eval runs for a citable success-rate number vs. the paper — only worth the Kaggle quota once A-D are solid.

## Verification Summary

| Phase | Artifact |
|---|---|
| A | `interact.py` completes a task; expert smoke run → PNGs + ~100% success rate |
| B | Kaggle base-VLM run → success rates over ≥5 trajectories; own ranking script matches `LlavaAgent.act()` |
| C | `diffusion_demo.py` plausible output; both `eval_reflect_vlm.sh` variants complete with logged success rates, compared against Phase B |
| D | Each `mini_roboworld/` module runs standalone; notes capture new understanding |
| E | Side-by-side frames; degraded-VLM comparison; optional full-scale metrics |

## Where `/graphify` Fits

Run `/graphify franka_vlm/workspace` (not `reflect-vlm-reference` — that's external reference material, not your own knowledge to graph) starting after Phase A once `notes/` and the first `mini_roboworld/` files exist. Re-run it as a phase-closing step after B, C, D (and E if done) — `workspace/` grows substantially each phase, so treat the graph rebuild as part of finishing each phase, not a one-time setup action.

## Critical Files Reference

- `roboworld/agent/llava.py` — `LlavaAgent.act()`, central to Phase B
- `run.py` — `imagine_with_sim`/`imagine_with_diffusion`/`main()`, central to Phase C
- `roboworld/agent/diffuser.py` — `DiffusionSim`, central to Phase C
- `roboworld/agent/utils.py` — `parse_act_txt`/`get_prompt`, Phase B & C
- `roboworld/constants.py`, `roboworld/envs/generator.py` — action vocabulary + task generation, Phase A & D
- `scripts/interact.py`, `scripts/eval_base_vlm.sh`, `scripts/eval_reflect_vlm.sh`, `scripts/eval_expert.sh` — runnable entrypoints for every reproduce milestone

## Open Gaps (verify, don't assume)

- Quantization flags not wired into `run.py`/eval scripts (Phase C — patch if Kaggle OOMs)
- `control.py` internals and `control_mode="script"` string (Phase A — grep to confirm)
- WSLg rendering `scripts/interact.py`'s GLFW window on the 8GB laptop (Phase A — PNG fallback ready)
- EGL rendering support in Kaggle's container for MuJoCo 3.1.2 (Phase B — `osmesa` fallback)
- Actual VRAM headroom for VLM+diffusion together on a 16GB Kaggle GPU (Phase C — sequential-load fallback if tight)
