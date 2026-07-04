"""Phase A dissection script: poke at the ReflectVLM task generator and env directly.

For each seed: call `generate_xml(seed)` and print what the procedural generator
produced (bodies, labels, dependency constraints, medium/hard level).

For one seed (--act-seed): build the real `FrankaAssemblyEnv`, drive it with manual
`act_txt` calls, and show the error codes each precondition violation returns.

Run headless (no GPU-heavy deps needed):
    python explore_env.py                    # 5 default seeds, act demo on the first
    python explore_env.py --seeds 1 2 3      # custom seeds
    python explore_env.py --no-act           # generator info only, much faster

On WSL2 also set LIBGL_ALWAYS_SOFTWARE=1 — the d3d12 GPU GL driver crashes under
sustained offscreen rendering (see notes/phase-a-env-and-sim.md).
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

# Must be set before mujoco is imported (transitively via roboworld).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REFERENCE_ROOT = Path(
    os.environ.get("REFLECT_VLM_ROOT", Path(__file__).resolve().parents[2] / "reflect-vlm-reference")
)
sys.path.insert(0, str(REFERENCE_ROOT))

from roboworld.envs.generator import generate_xml  # noqa: E402
from roboworld.envs.asset_path_utils import full_path_for  # noqa: E402


def describe_seed(seed):
    """Generate a task for `seed` and print the env_info the generator returns."""
    xml, info = generate_xml(seed=seed)
    n_bodies = info["n_bodies"]
    # Same cutoff run.py's build_env uses: boards with more than 5 bodies are "hard".
    level = "hard" if n_bodies > 5 else "medium"

    peg_names = [f"brick_{j}" for j in range(2, n_bodies + 1)]  # brick_1 is the board
    peg_descriptions = [info["brick_descriptions"][name] for name in peg_names]
    peg_labels = [desc.split()[0] for desc in peg_descriptions]  # first word = color
    id_to_label = {j: peg_labels[j - 2] for j in range(2, n_bodies + 1)}

    print(f"\n=== seed {seed} ===")
    print(f"n_bodies: {n_bodies} (board + {n_bodies - 1} pegs) -> level: {level}")
    print(f"board: {info['brick_descriptions']['brick_1']}")
    for name, desc, label in zip(peg_names, peg_descriptions, peg_labels):
        print(f"  {name}: {desc!r} (label: {label})")
    if info["dependencies"]:
        print("dependencies (u must be inserted before v):")
        for (u, v) in sorted(info["dependencies"]):
            print(f"  brick_{u} ({id_to_label[u]}) -> brick_{v} ({id_to_label[v]})")
    else:
        print("dependencies: none (any insertion order works)")

    return xml, info, peg_names, peg_descriptions, peg_labels


def act_demo(xml, info, peg_names, peg_descriptions, out_dir, reset_seed=1):
    """Build the real env and demonstrate act_txt error codes with manual calls."""
    from PIL import Image
    from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv, AssemblyOracle
    from roboworld.agent.oracle import OracleAgent

    xml_path = full_path_for(f"tmp_mini_{uuid.uuid4().hex[:8]}.xml")
    xml.write_to_file(xml_path)
    try:
        env = FrankaAssemblyEnv(
            board_name="brick_1", fixture_name=None,
            peg_names=peg_names, peg_descriptions=peg_descriptions,
            render_mode="offscreen", frame_skip=20, model_name=xml_path,
            max_episode_length=50000, magic_attaching=True,
        )
        oracle = AssemblyOracle(
            env=env, brick_ids=[j for j in range(2, info["n_bodies"] + 1)],
            brick_descriptions=peg_descriptions, dependencies=info["dependencies"],
        )
        oracle_agent = OracleAgent(oracle)
        env.reset(seed=reset_seed)

        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.goal_images["table_back"]).save(out_dir / "goal.png")
        Image.fromarray(env.read_pixels(camera_name="table_back")).save(out_dir / "initial.png")
        print(f"saved goal.png / initial.png to {out_dir}")

        print("peg colors as the env sees them:", env.peg_colors)
        oracle.update_state_from_env()
        print("oracle brick states:", oracle.get_states())

        # Let the oracle pick a feasible first target, then drive act_txt manually
        # around it to observe every error code.
        first = oracle_agent.act()  # e.g. "pick up red"
        color = first.split()[-1]
        other = next(c for c in env.peg_colors if c != color)
        print(f"oracle suggests: {first!r} -> demoing manual act_txt around color {color!r}")

        def try_act(txt):
            err = env.act_txt(txt)
            print(f"  act_txt({txt!r}) -> err={err}")
            return err

        print("expect -1 (not in hand):")
        try_act(f"insert {color}")
        print("expect 0 (valid pick up):")
        try_act(f"pick up {color}")
        print("expect -2 (already in hand):")
        try_act(f"pick up {color}")
        print("expect -1 (hands full with another object):")
        try_act(f"pick up {other}")
        print("expect 0 if insertable now (oracle said so), else watch the message:")
        try_act(f"insert {color}")

        brick = peg_names[env.peg_colors.index(color)]
        print(f"object_is_success({brick}/{color}):", env.object_is_success(brick))
        print("is_success (whole board):", env.is_success())
        Image.fromarray(env.read_pixels(camera_name="table_back")).save(out_dir / "after_insert.png")
        print(f"saved after_insert.png to {out_dir}")
        env.close()
    finally:
        os.remove(xml_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000000, 1000005)))
    parser.add_argument("--act-seed", type=int, default=None,
                        help="seed to run the act_txt demo on (default: first of --seeds)")
    parser.add_argument("--no-act", action="store_true", help="skip the act_txt demo (generator info only)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()

    act_seed = args.act_seed if args.act_seed is not None else args.seeds[0]
    seeds = list(args.seeds)
    if not args.no_act and act_seed not in seeds:
        seeds.append(act_seed)  # otherwise --act-seed outside --seeds would silently no-op
    for seed in seeds:
        xml, info, peg_names, peg_descriptions, peg_labels = describe_seed(seed)
        if not args.no_act and seed == act_seed:
            act_demo(xml, info, peg_names, peg_descriptions, args.out_dir / str(seed))


if __name__ == "__main__":
    main()
