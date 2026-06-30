"""Faithful LIBERO success-rate eval for the deployed pi0.5 policy server.

Two purposes:
  1. Reproduce OpenPI's in-distribution number (--prompt-mode canonical) — same tasks
     the model was trained on, from the tasks' init states. This mirrors
     examples/libero/main.py exactly (object settle, preprocessing, replan).
  2. Measure language generalization (--prompt-mode paraphrase) — same scenes & init
     states, but the instruction is a paraphrase. Use --prompt-mode both to run them
     paired (same init state for canonical and paraphrase) so wording is the only变量.

Run in the LIBERO client venv against a running policy server
(scripts/serve_policy.py --env LIBERO). Pause/stop the web UI first so it doesn't
contend with the eval for the single-threaded policy server.

  python libero_eval.py --suite libero_goal --trials 5 --prompt-mode both \
      --paraphrases paraphrases_libero_goal.json --out results_goal.json
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

DUMMY = [0.0] * 6 + [-1.0]
ENV_RES = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}


def _q2aa(q):
    q = list(q)
    if q[3] > 1.0:
        q[3] = 1.0
    if q[3] < -1.0:
        q[3] = -1.0
    d = math.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(d, 0.0) else (np.array(q[:3]) * 2.0 * math.acos(q[3])) / d


def _obs_element(obs, prompt, size):
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), size, size))
    wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), size, size))
    state = np.concatenate((obs["robot0_eef_pos"], _q2aa(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wri,
            "observation/state": state, "prompt": str(prompt)}


def rollout(env, init_state, prompt, client, max_steps, settle, replan, size):
    """One episode; returns True if the task's success predicate fires."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(settle):              # let dropped objects settle (benchmark parity)
        obs, _, _, _ = env.step(DUMMY)
    plan = collections.deque()
    for _ in range(max_steps):
        if not plan:
            chunk = client.infer(_obs_element(obs, prompt, size))["actions"]
            plan.extend(chunk[:replan])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            return True
    return False


def main():
    p = argparse.ArgumentParser(description="LIBERO eval: in-distribution vs paraphrase")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--trials", type=int, default=5, help="rollouts per task (benchmark uses 50)")
    p.add_argument("--max-tasks", type=int, default=0, help="0 = all tasks in the suite")
    p.add_argument("--task-ids", default="", help="comma-separated task ids; overrides --max-tasks")
    p.add_argument("--prompt-mode", choices=["canonical", "paraphrase", "both"], default="canonical")
    p.add_argument("--paraphrases", default="", help="JSON {canonical_instruction: paraphrase}")
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="")
    args = p.parse_args()

    para = {}
    if args.paraphrases:
        para = json.load(open(args.paraphrases))
    modes = ["canonical", "paraphrase"] if args.prompt_mode == "both" else [args.prompt_mode]

    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.task_ids.strip():
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip() != ""]
    else:
        n_all = suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, suite.n_tasks)
        task_ids = list(range(n_all))
    n = len(task_ids)
    max_steps = MAX_STEPS.get(args.suite, 300)

    totals = {m: [0, 0] for m in modes}   # mode -> [successes, trials]
    per_task = []
    for tid in task_ids:
        task = suite.get_task(tid)
        canon = str(task.language)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        prompts = {"canonical": canon, "paraphrase": para.get(canon, canon)}
        if "paraphrase" in modes and canon not in para:
            print("  [warn] no paraphrase for %r — falling back to canonical" % canon)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
        env.seed(args.seed)
        init_states = suite.get_task_init_states(tid)
        succ = {m: 0 for m in modes}
        for trial in range(args.trials):
            init = init_states[trial % len(init_states)]
            for m in modes:
                ok = rollout(env, init, prompts[m], client, max_steps, args.settle, args.replan, args.resize)
                succ[m] += int(ok)
                totals[m][0] += int(ok)
                totals[m][1] += 1
        env.close()
        row = {"task_id": tid, "canonical": canon, "paraphrase": prompts["paraphrase"],
               "success": {m: "%d/%d" % (succ[m], args.trials) for m in modes}}
        per_task.append(row)
        print("task %2d  %s  | %s" % (tid, "  ".join("%s %d/%d" % (m, succ[m], args.trials) for m in modes), canon[:50]))

    print("\n=== %s · %d tasks · %d trials/task ===" % (args.suite, n, args.trials))
    summary = {}
    for m in modes:
        s, t = totals[m]
        summary[m] = round(100.0 * s / t, 1) if t else 0.0
        print("  %-11s success: %.1f%%  (%d/%d)" % (m, summary[m], s, t))

    if args.out:
        json.dump({"suite": args.suite, "trials": args.trials, "tasks": n,
                   "summary_pct": summary, "per_task": per_task,
                   "when": datetime.datetime.now().isoformat(timespec="seconds")},
                  open(args.out, "w"), indent=2)
        print("saved %s" % args.out)


if __name__ == "__main__":
    main()
