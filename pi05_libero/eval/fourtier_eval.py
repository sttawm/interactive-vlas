#!/usr/bin/env python3
"""Four-tier eval for pi0.5 on LIBERO: original / natural / adversarial / oracle.

Per task, against a live policy server (scripts/serve_policy.py --env LIBERO):
  1. TIERS -- original at 10 trials (init states 0-9); each of 5 natural and 5
     adversarial phrases (fourtier_phrases.json) at 5 trials (inits 0-4, paired).
  2. ORACLE -- rollout board search, ported from phrase-rl scripts/search_boards.py
     (user spec 2026-07-29): board of 16 incl. the canonical + the 5 naturals
     (their screen cells reuse tier episodes), Gemini tops up to 16; screen = 5
     trials on inits 0-4; keep top-4; Gemini writes board-16-minus-keep new
     phrases seeing the ranked board (image-conditioned, gemini-3.5-flash,
     temp 0.9); converge when the best screen score stops improving (round >= 2)
     or the kept set saturates the screen; max 4 rounds.
  3. CONFIRM -- top-4 + canonical at 10 trials on VIRGIN inits 20-29. The
     confirm winner is the reported oracle number (guards winner's curse).

Every episode is one JSONL row in --out; resume replays the log and skips
completed (arm, phrase, init) cells, so kills are free. Board search state is
reconstructed from logged screen episodes; Gemini-minted board members are
persisted in a sidecar {out}.boards.json so a resume never re-mints a board.

Run in the LIBERO client venv (py3.8):
  PYTHONPATH=/workspace/openpi/third_party/libero MUJOCO_GL=egl \
    python fourtier_eval.py --shard lb1 --out /workspace/fourtier_lb1.jsonl
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime
import io
import json
import math
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pathlib
import urllib.request

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

DUMMY = [0.0] * 6 + [-1.0]
ENV_RES = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}

SHARDS = {  # interleaved so no pod carries all the slow libero_90 failures
    "lb1": [("libero_goal", 0), ("libero_goal", 1), ("libero_goal", 2),
            ("libero_90", 31), ("libero_90", 35)],
    "lb2": [("libero_goal", 3), ("libero_goal", 4), ("libero_goal", 5),
            ("libero_90", 44), ("libero_90", 77)],
    "lb3": [("libero_goal", 6), ("libero_goal", 7),
            ("libero_90", 14), ("libero_90", 54), ("libero_90", 70)],
    "lb4": [("libero_goal", 8), ("libero_goal", 9),
            ("libero_90", 64), ("libero_90", 12), ("libero_90", 28)],
    # E2 coverage extension: the remaining 10 mid-band libero_90 tasks
    # (incl. deliberate same-string/different-scene pairs 2/29 and 79/82)
    "lb1x": [("libero_90", 10), ("libero_90", 29), ("libero_90", 38)],
    "lb2x": [("libero_90", 79), ("libero_90", 2)],
    "lb3x": [("libero_90", 19), ("libero_90", 57), ("libero_90", 59)],
    "lb4x": [("libero_90", 60), ("libero_90", 82)],
}

# E1 deepen extension: extra virgin init windows on already-run tasks
DEEPEN_ORIG_INITS = list(range(10, 30))     # orig n 10 -> 30
DEEPEN_TIER_INITS = list(range(5, 10))      # each nat/adv phrase n 5 -> 10
DEEPEN_CONFIRM_INITS = list(range(30, 50))  # finalists n 10 -> 30 (2nd virgin window)

N_NAT = N_ADV = 5
ORIG_TRIALS, TIER_TRIALS = 10, 5
SCREEN_INITS = [0, 1, 2, 3, 4]   # 5-trial screen (user 2026-08-25); canonical+naturals reuse tier episodes
CONFIRM_INITS = list(range(20, 30))
BOARD, KEEP, MAX_ROUNDS = 16, 4, 4

GEN_PROMPT = """The attached image is a robot arm's camera view (LIBERO tabletop manipulation). We are searching for the instruction phrasing a trained robot policy follows most reliably.

Task instruction (original): "{nominal}"

Ranked results so far (successes out of {k} tries; HIGHER = better):
{board}

Write {n} NEW phrasings informed by what is winning: keep the elements that work (object names, adjectives, structure), vary what might improve. Use common household object names matching the image; concrete visible colors; imperative; under 15 words; same task meaning. No duplicates of the listed phrases.
Output exactly {n} lines, one per line, no numbering, no quotes."""


def _q2aa(q):
    q = list(q)
    q[3] = max(-1.0, min(1.0, q[3]))
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
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(settle):
        obs, _, _, _ = env.step(DUMMY)
    plan = collections.deque()
    for step in range(max_steps):
        if not plan:
            chunk = client.infer(_obs_element(obs, prompt, size))["actions"]
            plan.extend(chunk[:replan])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            return True, step + 1
    return False, max_steps


def gemini(model, parts, api_key, temperature=0.9, max_tokens=2000, retries=6):
    """REST generateContent (libero venv is py3.8; no google-genai SDK). Key in
    header only -- never in the URL or argv."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
    body = json.dumps({"contents": [{"parts": parts}],
                       "generationConfig": {"temperature": temperature,
                                            "maxOutputTokens": max_tokens}}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "x-goog-api-key": api_key})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            return "".join(p.get("text", "") for c in resp.get("candidates", [])
                           for p in c.get("content", {}).get("parts", []))
        except Exception as e:
            if attempt == retries - 1:
                raise
            print("  gemini retry %d: %s" % (attempt + 1, type(e).__name__), flush=True)
            time.sleep(min(5 * 2 ** attempt, 60))


def parse_lines(out, seen, n):
    res = []
    for line in out.splitlines():
        p = line.strip().strip('"').lstrip("-• ").strip()
        if p and 3 <= len(p.split()) <= 18 and p.lower() not in seen and len(res) < n:
            seen.add(p.lower())
            res.append(p)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True, choices=sorted(SHARDS))
    p.add_argument("--out", required=True)
    p.add_argument("--phrases", default=str(pathlib.Path(__file__).parent / "fourtier_phrases.json"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--phase", choices=["main", "deepen"], default="main",
                   help="deepen = E1: extend orig/tier/confirm cells on virgin inits, no board search")
    args = p.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        for line in open(os.path.expanduser("~/.bashrc")):
            if line.startswith("export GEMINI_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not api_key and not args.skip_oracle:
        raise SystemExit("GEMINI_API_KEY missing (needed for oracle board search)")

    phrases = json.load(open(args.phrases))
    boards_path = args.out + ".boards.json"
    boards_state = json.load(open(boards_path)) if os.path.exists(boards_path) else {}

    done = {}   # (suite, tid, phrase, init) -> success
    confirm_phrases = {}   # (suite, tid) -> set of confirm-arm phrases (for --phase deepen)
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[(r["suite"], r["task_id"], r["phrase"], r["init"])] = r["success"]
                if r["arm"] == "confirm":
                    confirm_phrases.setdefault((r["suite"], r["task_id"]), set()).add(r["phrase"])
            except (json.JSONDecodeError, KeyError):
                pass  # truncated tail row from a mid-write kill
        print("[resume] %d episodes already logged" % len(done), flush=True)

    logf = open(args.out, "a")
    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    suites = {}

    def run_cell(env, init_states, suite, tid, canon, arm, phrase, inits, max_steps):
        """Roll (phrase, init) pairs not yet logged; return successes over ALL inits."""
        succ = 0
        for ii in inits:
            key = (suite, tid, phrase, ii)   # arm-agnostic: (phrase, init) is the measurement
            if key in done:
                succ += int(done[key])
                continue
            t0 = time.time()
            ok, steps = rollout(env, init_states[ii], phrase, client, max_steps,
                                args.settle, args.replan, args.resize)
            done[key] = int(ok)
            succ += int(ok)
            logf.write(json.dumps({"suite": suite, "task_id": tid, "canonical": canon,
                                   "arm": arm, "phrase": phrase, "init": ii,
                                   "success": int(ok), "steps": steps,
                                   "sec": round(time.time() - t0, 1),
                                   "ts": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
            logf.flush()
        return succ

    for suite_name, tid in SHARDS[args.shard]:
        if suite_name not in suites:
            suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
        suite = suites[suite_name]
        task = suite.get_task(tid)
        canon = str(task.language)
        tiers = phrases[canon]
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        max_steps = MAX_STEPS.get(suite_name, 300)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES,
                                 camera_widths=ENV_RES)
        env.seed(args.seed)
        init_states = suite.get_task_init_states(tid)
        assert len(init_states) >= 30, "task %d has only %d init states" % (tid, len(init_states))
        t_task = time.time()
        print("== %s task %d: %r (%d inits)" % (suite_name, tid, canon, len(init_states)), flush=True)

        if args.phase == "deepen":
            s0 = run_cell(env, init_states, suite_name, tid, canon, "orig", canon,
                          DEEPEN_ORIG_INITS, max_steps)
            print("  deepen orig +%d inits -> %d succ" % (len(DEEPEN_ORIG_INITS), s0), flush=True)
            for kind, tag in (("natural", "nat"), ("adversarial", "adv")):
                for i, ph in enumerate(tiers[kind]):
                    run_cell(env, init_states, suite_name, tid, canon, "%s%d" % (tag, i + 1),
                             ph, DEEPEN_TIER_INITS, max_steps)
            for ph in sorted(confirm_phrases.get((suite_name, tid), [])):
                c = run_cell(env, init_states, suite_name, tid, canon, "confirm2", ph,
                             DEEPEN_CONFIRM_INITS, max_steps)
                print("  deepen confirm2 %d/%d  %r" % (c, len(DEEPEN_CONFIRM_INITS), ph[:60]), flush=True)
            env.close()
            print("  DEEPEN done %s/%d  [%.0fs]" % (suite_name, tid, time.time() - t_task), flush=True)
            continue

        # ---- 1. tiers -------------------------------------------------------
        s = run_cell(env, init_states, suite_name, tid, canon, "orig", canon,
                     list(range(ORIG_TRIALS)), max_steps)
        print("  orig %d/%d" % (s, ORIG_TRIALS), flush=True)
        for kind, tag in (("natural", "nat"), ("adversarial", "adv")):
            for i, ph in enumerate(tiers[kind]):
                s = run_cell(env, init_states, suite_name, tid, canon, "%s%d" % (tag, i + 1),
                             ph, list(range(TIER_TRIALS)), max_steps)
                print("  %s%d %d/%d  %r" % (tag, i + 1, s, TIER_TRIALS, ph[:60]), flush=True)

        if args.skip_oracle:
            env.close()
            continue

        # ---- 2. oracle board search ----------------------------------------
        bkey = "%s/%d" % (suite_name, tid)
        st = boards_state.setdefault(bkey, {"members": [], "rounds_done": 0, "converged": False})

        def screen_score(ph):
            return run_cell(env, init_states, suite_name, tid, canon, "screen", ph,
                            SCREEN_INITS, max_steps)

        # canonical + naturals seed the board; their screen cells reuse tier episodes
        seen = set()
        board = []
        for ph in [canon] + tiers["natural"] + st["members"]:
            if ph.lower() not in seen:
                seen.add(ph.lower())
                board.append(ph)

        env.reset()
        obs = env.set_init_state(init_states[0])
        for _ in range(args.settle):
            obs, _, _, _ = env.step(DUMMY)
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).save(buf, "PNG")
            img_part = {"inline_data": {"mime_type": "image/png",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}
        except Exception as e:
            print("  frame render failed (%s) -- text-only board gen" % type(e).__name__, flush=True)
            img_part = None

        def gen_new(n, board_txt):
            parts = ([img_part] if img_part else []) + [{"text": GEN_PROMPT.format(
                nominal=canon, k=len(SCREEN_INITS), board=board_txt, n=n)}]
            out = gemini("gemini-3.5-flash", parts, api_key)
            return parse_lines(out, seen, n)

        if len(board) < BOARD:
            board += gen_new(BOARD - len(board), "(no results yet -- first round)")
            st["members"] = board[:]
            json.dump(boards_state, open(boards_path, "w"), indent=1)

        scored = {ph: screen_score(ph) for ph in board}
        best_hist = []
        rd = st["rounds_done"] or 1
        while not st["converged"]:
            ranked = sorted(scored.items(), key=lambda x: -x[1])
            top = ranked[:KEEP]
            best_hist.append(top[0][1])
            print("  board r%d: best %d/%d  %r" % (rd, top[0][1], len(SCREEN_INITS), top[0][0][:60]), flush=True)
            if (len(best_hist) >= 2 and best_hist[-1] - best_hist[-2] < 1) \
               or all(v == len(SCREEN_INITS) for _, v in top) or rd >= MAX_ROUNDS:
                st["converged"] = True
                break
            btxt = "\n".join('%d. "%s"  %d/%d' % (i + 1, ph, v, len(SCREEN_INITS))
                             for i, (ph, v) in enumerate(ranked[:8]))
            newp = gen_new(BOARD - KEEP, btxt)
            if not newp:
                st["converged"] = True
                break
            st["members"] += newp
            rd += 1
            st["rounds_done"] = rd
            json.dump(boards_state, open(boards_path, "w"), indent=1)
            for ph in newp:
                scored[ph] = screen_score(ph)
        json.dump(boards_state, open(boards_path, "w"), indent=1)

        # ---- 3. confirm on virgin inits ------------------------------------
        finalists = [ph for ph, _ in sorted(scored.items(), key=lambda x: -x[1])[:KEEP]]
        if canon not in finalists:
            finalists.append(canon)
        conf = {ph: run_cell(env, init_states, suite_name, tid, canon, "confirm", ph,
                             CONFIRM_INITS, max_steps) for ph in finalists}
        ranked = sorted(conf.items(), key=lambda x: (-x[1], x[0] != canon))
        print("  ORACLE %s/%d: winner %d/10 %r (canon %d/10)  [%.0fs]" % (
            suite_name, tid, ranked[0][1], ranked[0][0][:60], conf.get(canon, -1),
            time.time() - t_task), flush=True)
        env.close()

    logf.close()
    print("SHARD-COMPLETE %s" % args.shard, flush=True)


if __name__ == "__main__":
    main()
