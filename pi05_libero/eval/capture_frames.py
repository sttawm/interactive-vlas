#!/usr/bin/env python3
"""Save the t=0 agentview frame (init state 0, seed 7, settled) for a list of
LIBERO tasks — scene images for phrase-rl's image-conditioned trace generation.

  MUJOCO_GL=egl python3 capture_frames.py --tasks frame_tasks.json --out /workspace/frames
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RES = 256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.load(open(args.tasks))
    suites = {}
    for t in tasks:
        suite, tid = t["suite"], t["task_id"]
        dst = out / f"{suite}__{tid:02d}.png"
        if dst.exists():
            continue
        if suite not in suites:
            suites[suite] = benchmark.get_benchmark_dict()[suite]()
        task = suites[suite].get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                 camera_heights=RES, camera_widths=RES)
        env.seed(args.seed)
        env.reset()
        init_states = suites[suite].get_task_init_states(tid)
        obs = env.set_init_state(init_states[0])
        for _ in range(args.settle):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        img = obs["agentview_image"][::-1]          # LIBERO renders upside-down
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(dst)
        env.close()
        print(f"saved {dst.name}", flush=True)
    print("FRAMES-DONE", flush=True)


if __name__ == "__main__":
    main()
