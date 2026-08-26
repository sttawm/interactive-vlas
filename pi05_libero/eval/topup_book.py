#!/usr/bin/env python3
"""Top up the book canonical's trained-scene cell from n=5 to n=20 for the
paper figure: roll "pick up the book and place it in the back compartment of
the caddy" in its libero_10 scene on init states 5-19 (the cross phase covered
0-4). Appends standard episode rows to /workspace/fourtier_topup.jsonl.

Run in the LIBERO client venv with the policy server up:
  PYTHONPATH=$LIBERO_PYTHONPATH MUJOCO_GL=egl python topup_book.py
"""
import datetime
import json
import pathlib
import time

from fourtier_eval import (DUMMY, ENV_RES, MAX_STEPS, rollout, _wcp,
                           benchmark, get_libero_path, OffScreenRenderEnv)

BOOK = "pick up the book and place it in the back compartment of the caddy"
OUT = "/workspace/fourtier_topup.jsonl"

suite = benchmark.get_benchmark_dict()["libero_10"]()
tid = next(i for i in range(suite.n_tasks) if str(suite.get_task(i).language) == BOOK)
task = suite.get_task(tid)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
env.seed(7)
init_states = suite.get_task_init_states(tid)
client = _wcp.WebsocketClientPolicy("127.0.0.1", 8000)
ms = MAX_STEPS["libero_10"]
with open(OUT, "a") as f:
    for ii in range(5, 20):
        t0 = time.time()
        ok, steps = rollout(env, init_states[ii], BOOK, client, ms, 10, 5, 224)
        f.write(json.dumps({"suite": "libero_10", "task_id": tid, "canonical": BOOK,
                            "arm": "cross", "phrase": BOOK, "init": ii,
                            "success": int(ok), "steps": steps,
                            "sec": round(time.time() - t0, 1),
                            "ts": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
        f.flush()
        print("init %d: %s" % (ii, "OK" if ok else "fail"), flush=True)
env.close()
print("TOPUP-COMPLETE")
