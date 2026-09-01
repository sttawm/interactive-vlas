#!/usr/bin/env python3
"""Generic queue runner for the pi0.5/LIBERO phrase bank.

Consumes an ORDERED queue json (built Mac-side, categories interleaved so an
early stop preserves the canonical/natural/adversarial distribution) and rolls
each (suite, task_id, phrase, init) not yet logged. Same jsonl row format as
fourtier_eval.py, so the phrase-rl seed-bank converter ingests it unchanged.

Run in the LIBERO client venv (py3.8), against a live policy server:

  MUJOCO_GL=egl python3 bank_eval.py --queue queue_lb1.json --out bank_lb1.jsonl
"""
import argparse
import datetime
import json
import os
import time

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _wcp
import pathlib

from fourtier_eval import rollout, ENV_RES, MAX_STEPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    queue = json.load(open(args.queue))
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["suite"], r["task_id"], r["phrase"], r["init"]))
            except (json.JSONDecodeError, KeyError):
                pass
        print("[resume] %d episodes already logged" % len(done), flush=True)

    logf = open(args.out, "a")
    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    suites = {}
    cur = {"key": None, "env": None, "inits": None}   # one env at a time

    def env_for(suite, tid):
        if cur["key"] == (suite, tid):
            return cur["env"], cur["inits"]
        if cur["env"] is not None:
            cur["env"].close()
        if suite not in suites:
            suites[suite] = benchmark.get_benchmark_dict()[suite]()
        task = suites[suite].get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                 camera_heights=ENV_RES, camera_widths=ENV_RES)
        env.seed(args.seed)
        cur.update(key=(suite, tid), env=env,
                   inits=suites[suite].get_task_init_states(tid))
        return cur["env"], cur["inits"]

    total = sum(len(it["inits"]) for it in queue)
    done_n = 0
    for it in queue:
        suite, tid = it["suite"], it["task_id"]
        max_steps = max(MAX_STEPS.get(suite, 300), 300)
        for ii in it["inits"]:
            key = (suite, tid, it["phrase"], ii)
            if key in done:
                done_n += 1
                continue
            env, init_states = env_for(suite, tid)
            t0 = time.time()
            ok, steps = rollout(env, init_states[ii], it["phrase"], client,
                                max_steps, args.settle, args.replan, args.resize)
            done.add(key)
            done_n += 1
            logf.write(json.dumps({
                "suite": suite, "task_id": tid, "canonical": it["canonical"],
                "arm": it["arm"], "phrase": it["phrase"], "init": ii,
                "success": int(ok), "steps": steps,
                "sec": round(time.time() - t0, 1),
                "ts": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
            logf.flush()
        if done_n % 20 < len(it["inits"]):
            print("progress %d/%d episodes" % (done_n, total), flush=True)
    if cur["env"] is not None:
        cur["env"].close()
    logf.close()
    print("QUEUE-COMPLETE %d/%d" % (done_n, total), flush=True)


if __name__ == "__main__":
    main()
