"""pi0.5 + LIBERO backend for the shared interactive web UI.

This is the LIBERO/pi0.5-specific half: it owns the simulator, the policy client, and
the rollout thread, and implements the worker contract that `shared/webui.py` renders.
All UI / routing / video composition is generic and lives in shared.webui.

Reuses OpenPI's official LIBERO observation/action handling (examples/libero/main.py);
the only real change is that the prompt is a live, user-controlled variable. Runs in the
LIBERO client venv (Python 3.8) and talks to an OpenPI policy server over websocket.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import math
import os
import pathlib
import re
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# Make the repo-root `shared` package importable when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shared import webui  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pi05_libero")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial", "libero_90"]
MAX_STEPS_BY_SUITE = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
                      "libero_10": 520, "libero_90": 400}


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoWorker(threading.Thread):
    """Implements the shared.webui worker contract for pi0.5 + LIBERO."""

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        self._benchmark_dict = benchmark.get_benchmark_dict()
        self._task_cache = {}  # suite -> [task language, ...]
        self._scene_cache = {}  # suite -> [(scene label, [task ids]), ...]

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._instruction = ""
        self._latest_jpeg = _placeholder_jpeg("starting...")
        self._paused = True
        self._reset_to = None
        self._clear_plan = False
        self._dbg_last_prompt = None

        self._st = {  # raw status fields; snapshot_status() formats these for display
            "suite": SUITES[0], "task_id": 0, "task_language": "", "instruction": "",
            "sent_prompt": "", "sent_step": 0, "step": 0,
            "step_limit": args.max_rollout_steps, "paused": True, "limit_reached": False,
            "success": False, "connected": False,
        }
        self._record_frames = []
        self._record_prompts = []
        self._run_dir = None

    # ----- task metadata (cheap; no env) -----

    def _task_languages(self, suite):
        if suite not in self._task_cache:
            ts = self._benchmark_dict[suite]()
            self._task_cache[suite] = [ts.get_task(i).language for i in range(ts.n_tasks)]
        return self._task_cache[suite]

    # ----- contract: config / status / control -----

    def _objset(self, task):
        """The set of manipulable objects in a task's bddl — the true 'scene' key
        (tasks with identical object sets share a scene)."""
        txt = (pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file).read_text()
        m = re.search(r"\(:objects(.*?)\)\s*\(:", txt, re.S) or re.search(r"\(:objects(.*?)\)", txt, re.S)
        return frozenset(re.findall(r"([a-z_]+_\d+)\s", m.group(1))) if m else frozenset()

    def _scene_groups(self, suite):
        """Ordered [(label, [task_ids])] grouping tasks that share a scene, keyed by the
        bddl object set (verified criterion). Label = the bddl SCENE prefix when present
        (libero_10/90), else the instruction for a lone task, else 'Scene N'. Cached."""
        if suite in self._scene_cache:
            return self._scene_cache[suite]
        ts = self._benchmark_dict[suite]()
        langs = self._task_languages(suite)
        groups = collections.OrderedDict()  # objset -> [task_ids]
        prefix = {}
        for i in range(ts.n_tasks):
            t = ts.get_task(i)
            groups.setdefault(self._objset(t), []).append(i)
            m = re.match(r"([A-Z][A-Z_]*SCENE\d+)", t.bddl_file)
            prefix[i] = m.group(1) if m else None
        out, scene_n = [], 0
        for tids in groups.values():
            if prefix[tids[0]]:
                label = prefix[tids[0]]
            elif len(tids) == 1:
                label = langs[tids[0]]
            else:
                scene_n += 1
                label = "Scene %d" % scene_n
            out.append((label, tids))
        self._scene_cache[suite] = out
        return out

    def config(self):
        options_by = {}
        for s in SUITES:
            langs = self._task_languages(s)
            opts = []
            for label, tids in self._scene_groups(s):
                rep = tids[0]
                disp = label if len(tids) == 1 else "%s — %d tasks" % (label, len(tids))
                opts.append({"value": str(rep), "label": disp,
                             "default_prompt": langs[rep],
                             "examples": [langs[i] for i in tids]})
            options_by[s] = opts
        return {
            "title": "π0.5 · LIBERO",
            "selectors": [
                {"name": "suite", "label": "Suite", "options": SUITES},
                {"name": "scene", "label": "Scene / task", "depends_on": "suite", "options_by": options_by},
            ],
            "instruction_label": "Instruction to π0.5 — blank uses the scene's default task",
            "instruction_placeholder": "e.g. pick up the bowl",
        }

    def default_prompt(self, selection):
        try:
            suite = selection.get("suite", SUITES[0])
            tid = int(selection.get("scene", selection.get("task", 0)))
            return str(self._task_languages(suite)[tid])
        except Exception:
            return ""

    def set_instruction(self, text):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._instruction = text
            self._clear_plan = True
            self._st["instruction"] = text
        logger.info("Instruction set: %r", text)

    def request_reset(self, selection, instruction=""):
        instruction = (instruction or "").strip()
        canonical = self.default_prompt(selection)
        with self._lock:
            self._reset_to = dict(selection)
            self._instruction = instruction or canonical
            self._paused = True
            self._st["paused"] = True
            self._st["instruction"] = self._instruction

    def set_paused(self, paused):
        with self._lock:
            self._paused = bool(paused)
            self._st["paused"] = bool(paused)

    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def snapshot_status(self):
        with self._lock:
            s = dict(self._st)
        if not s["connected"]:
            state = "loading…"
        elif s["limit_reached"]:
            state = "⏹ step limit (%d) — Reset" % s["step_limit"]
        elif s["paused"]:
            state = "⏸ paused"
        else:
            state = "▶ running"
        prompt = s["instruction"] or "—"
        if s["instruction"] and s["instruction"] != s["sent_prompt"]:
            prompt += "   (pending — not executed yet)"
        sent = ("%s   (@step %d)" % (s["sent_prompt"], s["sent_step"])) if s["sent_prompt"] \
            else "— nothing sent yet (press Play) —"
        return {
            "Task": "%s · task %s" % (s["suite"], s["task_id"]),
            "Scene goal": s["task_language"] or "—",
            "Prompt (set)": prompt,
            "→ Sent to policy": sent,
            "Goal met (orig. task)": "yes ✓" if s["success"] else "no",
            "Step": "%d / %d" % (s["step"], s["step_limit"]),
            "State": state,
            # consumed by the frontend (Play button + limit handling), not shown as rows:
            "paused": s["paused"],
            "limit_reached": s["limit_reached"],
        }

    def save_video(self, name, speed=1.0):
        with self._lock:
            frames = list(self._record_frames)
            prompts = list(self._record_prompts)
        return webui.compose_video(frames, prompts, name, speed=speed, runs_dir=self.args.runs_dir)

    def stop(self):
        self._stop.set()

    # ----- recording -----

    def _start_new_run(self, suite, task_id):
        self._finalize_run()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = pathlib.Path(self.args.runs_dir) / ("%s_%s_t%s" % (stamp, suite, task_id))
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._record_frames = []
        self._record_prompts = []
        self._run_meta = {"suite": suite, "task_id": task_id, "started": stamp,
                          "openpi_commit": os.environ.get("OPENPI_COMMIT", "")}

    def _finalize_run(self):
        if not self._run_dir:
            return
        try:
            if self._record_frames:
                imageio.mimwrite(str(self._run_dir / "rollout.mp4"),
                                 [np.asarray(f) for f in self._record_frames], fps=10)
            with open(self._run_dir / "instructions.txt", "w") as fh:
                last = None
                for i, p in enumerate(self._record_prompts):
                    if p != last:
                        fh.write("step=%d\t%s\n" % (i, p))
                        last = p
            with open(self._run_dir / "metadata.json", "w") as fh:
                meta = dict(self._run_meta)
                meta["final_step"] = self._st["step"]
                json.dump(meta, fh, indent=2)
        except Exception:
            logger.exception("finalize_run failed")
        finally:
            self._run_dir = None

    # ----- the rollout loop -----

    def _make_env(self, suite, task_id):
        task_suite = self._benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION,
            # Interactive rollouts are open-ended; without ignore_done robosuite
            # terminates at horizon=1000 and the next step() raises. Success still
            # comes from LIBERO's _check_success(), unaffected.
            ignore_done=True, horizon=10_000_000,
        )
        env.seed(self.args.seed)
        return env, task.language, task_suite.get_task_init_states(task_id)

    def run(self):
        env = None
        obs = None
        action_plan = collections.deque()
        step = 0
        while not self._stop.is_set():
            with self._lock:
                reset_to = self._reset_to
                self._reset_to = None
                paused = self._paused
                instruction = self._instruction
                clear_plan = self._clear_plan
                self._clear_plan = False

            if reset_to is not None:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                suite = reset_to.get("suite", SUITES[0])
                # selection["scene"] is the representative task id for that scene.
                task_id = int(reset_to.get("scene", reset_to.get("task", 0)))
                init_id = int(reset_to.get("init", 0))
                logger.info("Loading %s task %s...", suite, task_id)
                try:
                    env, task_language, init_states = self._make_env(suite, task_id)
                    env.reset()
                    init_id = max(0, min(init_id, len(init_states) - 1))
                    obs = env.set_init_state(init_states[init_id])
                except Exception:
                    logger.exception("env load failed")
                    env = None
                    continue
                action_plan.clear()
                step = 0
                self._start_new_run(suite, task_id)
                with self._lock:
                    self._st.update(suite=suite, task_id=task_id, task_language=str(task_language),
                                    instruction=self._instruction, step=0, sent_prompt="",
                                    sent_step=0, success=False, limit_reached=False,
                                    max_steps=MAX_STEPS_BY_SUITE.get(suite, 300), connected=True)
                self._publish_frame(obs, self._instruction)
                continue

            if env is None or paused:
                time.sleep(0.05)
                continue
            with self._lock:
                if self._st["limit_reached"]:
                    time.sleep(0.05)
                    continue

            if clear_plan:
                action_plan.clear()
            try:
                obs = self._step_once(env, obs, action_plan, instruction, step)
                step += 1
            except Exception:
                logger.exception("step failed; pausing")
                self.set_paused(True)
        self._finalize_run()

    def _step_once(self, env, obs, action_plan, instruction, step):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, self.args.resize_size, self.args.resize_size))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, self.args.resize_size, self.args.resize_size))

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": np.concatenate((obs["robot0_eef_pos"],
                                                     _quat2axisangle(obs["robot0_eef_quat"]),
                                                     obs["robot0_gripper_qpos"])),
                "prompt": str(instruction),
            }
            sent = str(instruction)
            if sent != self._dbg_last_prompt:
                logger.info("POLICY PROMPT -> %r (canonical %r)", sent, self._st.get("task_language"))
                self._dbg_last_prompt = sent
            with self._lock:
                self._st["sent_prompt"] = sent
                self._st["sent_step"] = step + 1
            action_chunk = self.client.infer(element)["actions"]
            action_plan.extend(action_chunk[: self.args.replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        self._record_frames.append(img)
        self._record_prompts.append(str(instruction))
        self._publish_frame(obs, instruction)

        reached_limit = (step + 1) >= self.args.max_rollout_steps
        with self._lock:
            self._st["step"] = step + 1
            self._st["success"] = bool(done)  # momentary, non-terminal
            if reached_limit:
                self._st["limit_reached"] = True
                self._st["paused"] = True
                self._paused = True
        if reached_limit:
            logger.info("Reached step limit (%s)", self.args.max_rollout_steps)
            self._finalize_run()
        return obs

    def _publish_frame(self, obs, overlay=None):
        DS = 384
        frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        frame = cv2.resize(frame, (DS, DS), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if overlay:
            label = overlay if len(overlay) < 60 else overlay[:57] + "..."
            cv2.rectangle(frame, (0, 0), (DS, 24), (0, 0, 0), -1)
            cv2.putText(frame, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()


def _placeholder_jpeg(text):
    img = np.zeros((384, 384, 3), dtype=np.uint8)
    cv2.putText(img, text, (16, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def main():
    p = argparse.ArgumentParser(description="Interactive pi0.5 + LIBERO (shared web UI)")
    p.add_argument("--host", default="127.0.0.1", help="policy server host")
    p.add_argument("--port", type=int, default=8000, help="policy server port")
    p.add_argument("--web-port", type=int, default=8888, help="web UI port")
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--max-rollout-steps", type=int, default=5000)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--runs-dir", default="runs")
    # Accepted for back-compat with run.sh; LIBERO suite is chosen in the UI.
    p.add_argument("--task-suite-name", default="libero_10")
    args = p.parse_args()

    worker = LiberoWorker(args)
    worker.start()
    app = webui.build_app(worker)
    logger.info("Web UI on :%d (policy server %s:%d)", args.web_port, args.host, args.port)
    app.run(host="0.0.0.0", port=args.web_port, threaded=True)


if __name__ == "__main__":
    main()
