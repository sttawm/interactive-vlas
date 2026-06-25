"""Interactive LIBERO runner for pi0.5 (OpenPI).

Runs a LIBERO simulation whose language instruction you type from a browser, and
streams the rendered camera view back so you can watch the policy act in real
time. Instructions can be changed mid-rollout (corrections / staged subgoals).

This deliberately reuses OpenPI's official LIBERO observation/action handling
(see examples/libero/main.py) so behavior matches the benchmark eval. The only
real change is that the prompt is a live, user-controlled variable instead of the
fixed benchmark task description.

Runs inside the LIBERO client venv (Python 3.8). The pi0.5 policy itself runs in
a separate OpenPI policy server (scripts/serve_policy.py --env LIBERO), which this
talks to over websocket.
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
import threading
import time

# MuJoCo must render offscreen on a headless GPU box. EGL uses the GPU; override
# with MUJOCO_GL=osmesa for CPU rendering if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("interactive_libero")

# Matches OpenPI's LIBERO eval constants.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _quat2axisangle(quat):
    """Copied from OpenPI/robosuite to match training-time state preprocessing."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class RolloutWorker(threading.Thread):
    """Owns the LIBERO env in a single thread and steps pi0.5 against a live prompt.

    MuJoCo contexts are thread-affine, so the env is created and stepped only here.
    The Flask request threads communicate via the lock-protected fields below.
    """

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Shared state (guarded by _lock).
        self._instruction = ""
        self._latest_jpeg = _placeholder_jpeg("starting...")
        self._paused = True  # start paused until the user picks a task / hits go
        self._reset_to = None  # (suite, task_id, init_id) requested reset
        self._clear_plan = False  # force a replan now (e.g. after an instruction change)
        self._dbg_last_prompt = None  # log the prompt sent to the policy whenever it changes
        self._dbg_was_solved = False  # de-dupe the "goal satisfied" log line

        # Status (guarded by _lock).
        self.status = {
            "suite": args.task_suite_name,
            "task_id": args.task_id,
            "task_language": "",
            "instruction": "",
            "step": 0,
            "max_steps": MAX_STEPS_BY_SUITE.get(args.task_suite_name, 300),
            "step_limit": args.max_rollout_steps,
            "limit_reached": False,
            "paused": True,
            "done": False,
            "success": False,
            "sent_prompt": "",   # exact string fed to the policy on the last inference
            "sent_step": 0,
            "connected": False,
        }

        # Per-rollout recording (only touched by worker thread).
        self._record_frames = []
        self._record_actions = []
        self._record_prompts = []  # prompt active at each recorded frame (for captions)
        self._record_instructions = []  # (wall_time, step, instruction)
        self._run_dir = None

        # Build the benchmark dict once.
        self._benchmark_dict = benchmark.get_benchmark_dict()

    # ----- public API used by Flask handlers -----

    def set_instruction(self, text):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._instruction = text
            self._clear_plan = True  # react immediately to corrections
            self.status["instruction"] = text
        # Record outside the env loop; step recorded by worker.
        self._record_instructions.append(
            [datetime.datetime.now().isoformat(timespec="seconds"), self.status["step"], text]
        )
        logger.info("New instruction @step %s: %r", self.status["step"], text)

    def request_reset(self, suite, task_id, init_id, instruction=""):
        # Land paused & ready: the env loads and shows its first frame, but the
        # rollout doesn't step until the user presses Play.
        instruction = (instruction or "").strip()
        # Resolve the canonical goal now (cheap — task metadata, no env) and set the
        # prompt synchronously. Otherwise a Send arriving during the ~15s env load
        # would be clobbered when the slow reset path finally assigns the prompt.
        canonical = ""
        try:
            canonical = str(self._benchmark_dict[suite]().get_task(int(task_id)).language)
        except Exception:
            logger.exception("Could not resolve canonical goal for %s task %s", suite, task_id)
        with self._lock:
            self._reset_to = (suite, int(task_id), int(init_id))
            self._instruction = instruction or canonical
            self._paused = True
            self.status["paused"] = True
            self.status["instruction"] = self._instruction

    def set_paused(self, paused):
        with self._lock:
            self._paused = bool(paused)
            self.status["paused"] = bool(paused)

    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def snapshot_status(self):
        with self._lock:
            return dict(self.status)

    def stop(self):
        self._stop.set()

    # ----- worker internals -----

    def _make_env(self, suite, task_id):
        task_suite = self._benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        task_description = task.language
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
            # Interactive rollouts are open-ended, but robosuite terminates the
            # episode at horizon=1000 (then any further step() raises "executing
            # action in terminated episode"). ignore_done lets it run forever;
            # task success still comes from LIBERO's _check_success(), unaffected.
            ignore_done=True,
            horizon=10_000_000,
        )
        env.seed(self.args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        return env, task_description, init_states

    def _start_new_run(self, suite, task_id, task_language):
        self._finalize_run()  # flush any previous run
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = pathlib.Path(self.args.runs_dir) / f"{stamp}_{suite}_t{task_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._record_frames = []
        self._record_actions = []
        self._record_prompts = []
        self._record_instructions = []
        self._run_meta = {
            "suite": suite,
            "task_id": task_id,
            "task_language": task_language,
            "openpi_commit": os.environ.get("OPENPI_COMMIT", ""),
            "started": stamp,
            "replan_steps": self.args.replan_steps,
            "seed": self.args.seed,
        }
        logger.info("Recording run to %s", run_dir)

    def _finalize_run(self):
        if not self._run_dir:
            return
        try:
            if self._record_frames:
                imageio.mimwrite(
                    str(self._run_dir / "rollout.mp4"),
                    [np.asarray(f) for f in self._record_frames],
                    fps=10,
                )
            if self._record_actions:
                np.save(str(self._run_dir / "actions.npy"), np.asarray(self._record_actions))
            with open(self._run_dir / "instructions.txt", "w") as fh:
                for wall, step, text in self._record_instructions:
                    fh.write(f"{wall}\tstep={step}\t{text}\n")
            meta = dict(self._run_meta)
            with self._lock:
                meta["final_success"] = self.status["success"]
                meta["final_step"] = self.status["step"]
            meta["ended"] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            with open(self._run_dir / "metadata.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            logger.info("Saved run: %s (%d frames)", self._run_dir, len(self._record_frames))
        except Exception:  # don't let logging kill the worker
            logger.exception("Failed to finalize run")
        finally:
            self._run_dir = None  # guard against finalizing the same run twice

    def run(self):
        env = None
        task_language = ""
        init_states = None
        action_plan = collections.deque()
        step = 0

        # Wait for first reset request.
        while not self._stop.is_set():
            with self._lock:
                reset_to = self._reset_to
                self._reset_to = None
                paused = self._paused
                instruction = self._instruction
                clear_plan = self._clear_plan
                self._clear_plan = False

            if reset_to is not None:
                suite, task_id, init_id = reset_to
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                logger.info("Loading %s task %s (init %s)...", suite, task_id, init_id)
                env, task_language, init_states = self._make_env(suite, task_id)
                env.reset()
                init_id = max(0, min(init_id, len(init_states) - 1))
                obs = env.set_init_state(init_states[init_id])
                action_plan.clear()
                step = 0
                self._start_new_run(suite, task_id, task_language)
                # The prompt was set synchronously in request_reset (so a Send during
                # this slow env load isn't clobbered) — don't reassign it here.
                with self._lock:
                    self.status.update(
                        suite=suite,
                        task_id=task_id,
                        task_language=str(task_language),
                        instruction=self._instruction,
                        step=0,
                        max_steps=MAX_STEPS_BY_SUITE.get(suite, 300),
                        done=False,
                        success=False,
                        limit_reached=False,
                        sent_prompt="",
                        sent_step=0,
                        connected=True,
                    )
                self._record_instructions.append(
                    [datetime.datetime.now().isoformat(timespec="seconds"), 0, self._instruction]
                )
                self._publish_frame(obs)
                continue

            if env is None or paused:
                time.sleep(0.05)
                continue

            with self._lock:
                stopped = self.status["limit_reached"]  # solving no longer stops; only the step cap does
            if stopped:
                time.sleep(0.05)
                continue

            if clear_plan:
                action_plan.clear()

            try:
                obs = self._step_once(env, obs, action_plan, instruction, step)
                step += 1
            except Exception:
                logger.exception("Step failed; pausing rollout")
                self.set_paused(True)
                continue

        self._finalize_run()

    def _step_once(self, env, obs, action_plan, instruction, step):
        # Preprocess exactly like OpenPI's LIBERO eval (rotate 180, resize+pad).
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self.args.resize_size, self.args.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, self.args.resize_size, self.args.resize_size)
        )

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ),
                "prompt": str(instruction),
            }
            sent = str(instruction)
            if sent != self._dbg_last_prompt:
                logger.info("POLICY PROMPT -> %r (canonical goal is %r)",
                            sent, self.status.get("task_language"))
                self._dbg_last_prompt = sent
            # Record the EXACT string fed to the policy this inference (ground truth
            # for the UI, captured inside the infer path).
            with self._lock:
                self.status["sent_prompt"] = sent
                self.status["sent_step"] = step + 1
            action_chunk = self.client.infer(element)["actions"]
            action_plan.extend(action_chunk[: self.args.replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        self._record_frames.append(img)
        self._record_actions.append(np.asarray(action))
        self._record_prompts.append(str(instruction))
        self._publish_frame(obs, overlay=instruction, step=step + 1)

        reached_limit = (step + 1) >= self.args.max_rollout_steps
        with self._lock:
            self.status["step"] = step + 1
            # success is the CURRENT goal-satisfied state (non-terminal) — solving
            # the task no longer ends the rollout; experiment freely.
            self.status["success"] = bool(done)
            if reached_limit:
                self.status["limit_reached"] = True
                self.status["paused"] = True
                self._paused = True
        if done and not self._dbg_was_solved:
            logger.info("Goal satisfied at step %s (rollout continues)", step + 1)
            self._dbg_was_solved = True
        elif not done:
            self._dbg_was_solved = False
        if reached_limit:
            logger.info("Reached step limit (%s); stopping rollout", self.args.max_rollout_steps)
            self._finalize_run()
        return obs

    def _publish_frame(self, obs, overlay=None, step=None):
        # Show the model's-eye agentview (rotated to be right-side up). Kept small
        # (384px, q70 ~= 12KB) so the MJPEG stream stays smooth over distant links;
        # the browser upscales it to the 512px view box.
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

    def compose_video(self, name, speed=1.0, out_size=480):
        """Render the run-so-far to runs/videos/<name>.mp4 with the active prompt
        captioned below each frame, at fps = 10 x speed. Returns the path (or None
        if there are no frames yet)."""
        # Snapshot under lock (cheap — copies references, not pixels).
        with self._lock:
            frames = list(self._record_frames)
            prompts = list(self._record_prompts)
            goal = self.status.get("task_language", "")
        if not frames:
            return None
        if len(prompts) < len(frames):  # pad in case of a race on the last frame
            prompts += [prompts[-1] if prompts else ""] * (len(frames) - len(prompts))

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "video"
        vid_dir = pathlib.Path(self.args.runs_dir) / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        path = vid_dir / f"{safe}.mp4"

        cap_h = 64
        W, font = out_size, cv2.FONT_HERSHEY_SIMPLEX
        fps = max(1, int(round(10 * float(speed))))
        writer = imageio.get_writer(str(path), fps=fps, macro_block_size=None)
        try:
            for img, prompt in zip(frames, prompts):
                fr = cv2.resize(img, (W, out_size), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((out_size + cap_h, W, 3), dtype=np.uint8)
                canvas[:out_size] = fr  # img is RGB; imageio writes RGB
                lines = _wrap_text(prompt or "—", font, 0.5, 1, W - 16)[:2]
                y = out_size + 24
                for ln in lines:
                    cv2.putText(canvas, ln, (8, y), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
                    y += 22
                writer.append_data(canvas)
        finally:
            writer.close()
        logger.info("Saved video %s (%d frames, %dx, goal=%r)", path, len(frames), fps // 10 or 1, goal)
        return path


def _wrap_text(text, font, scale, thick, max_w):
    """Greedy word-wrap so a caption fits within max_w pixels."""
    out, cur = [], ""
    for w in str(text).split():
        test = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thick)
        if tw > max_w and cur:
            out.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        out.append(cur)
    return out or [""]


def _placeholder_jpeg(text):
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def list_tasks(suite):
    bd = benchmark.get_benchmark_dict()
    if suite not in bd:
        return []
    ts = bd[suite]()
    return [{"task_id": i, "language": ts.get_task(i).language} for i in range(ts.n_tasks)]


# ----- Flask app -----

def build_app(worker, args):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML, suite=args.task_suite_name)

    @app.route("/frame.jpg")
    def frame():
        return Response(worker.latest_jpeg(), mimetype="image/jpeg")

    @app.route("/stream.mjpg")
    def stream():
        # MJPEG multipart stream over one persistent connection: frame rate is
        # bandwidth-bound, not per-request-latency bound (matters a lot when the
        # client is far from the pod / behind a reverse proxy). The browser renders
        # multipart/x-mixed-replace natively in an <img>.
        def gen():
            last = None
            while True:
                jpg = worker.latest_jpeg()
                if jpg is not last:  # only push when the frame actually changes
                    last = jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.04)  # cap ~25 fps
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return jsonify(worker.snapshot_status())

    @app.route("/tasks")
    def tasks():
        suite = request.args.get("suite", args.task_suite_name)
        return jsonify({"suite": suite, "tasks": list_tasks(suite)})

    @app.route("/instruction", methods=["POST"])
    def instruction():
        worker.set_instruction((request.json or {}).get("text", ""))
        return jsonify(ok=True)

    @app.route("/reset", methods=["POST"])
    def reset():
        body = request.json or {}
        worker.request_reset(
            body.get("suite", args.task_suite_name),
            body.get("task_id", args.task_id),
            body.get("init_id", 0),
            instruction=body.get("instruction", ""),
        )
        return jsonify(ok=True)

    @app.route("/pause", methods=["POST"])
    def pause():
        worker.set_paused((request.json or {}).get("paused", True))
        return jsonify(ok=True)

    @app.route("/save_video", methods=["POST"])
    def save_video():
        body = request.json or {}
        name = body.get("name") or "rollout"
        try:
            speed = float(body.get("speed", 1) or 1)
        except (TypeError, ValueError):
            speed = 1.0
        path = worker.compose_video(name, speed=speed)
        if path is None:
            return jsonify(ok=False, error="No frames recorded yet — load a scene and press Play first."), 400
        # Saved on the pod under runs/videos/<name>.mp4; also stream it back as a download.
        # send_file requires an absolute path on Flask 3.x.
        path = pathlib.Path(path).resolve()
        return send_file(str(path), mimetype="video/mp4", as_attachment=True,
                         download_name=path.name)

    return app


INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Interactive LIBERO · pi0.5</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
 .wrap{max-width:900px;margin:0 auto}
 h1{font-size:18px;font-weight:600}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 #view{width:512px;height:512px;background:#000;border:1px solid #333;border-radius:8px}
 .panel{flex:1;min-width:300px}
 input,select,button,textarea{font-size:14px;padding:8px;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#eee}
 input[type=text]{width:100%}
 button{background:#2d6cdf;border:none;cursor:pointer}
 button.alt{background:#444}
 label{font-size:12px;color:#aaa;display:block;margin:14px 0 5px}
 .hint{color:#888;font-size:12px;margin-top:5px}
 button:active{transform:scale(0.98)} button:disabled{opacity:0.45;cursor:default}
 .green{background:#2ea043} #play.ready{background:#2ea043}
 button.big{padding:11px;font-weight:600;font-size:15px}
 .toast{min-height:18px;font-size:13px;color:#3fb950;margin:12px 0 6px;transition:opacity .25s;opacity:0}
 .toast.show{opacity:1}
 .status{background:#000;padding:4px 12px;border-radius:8px;font-size:13px}
 .srow{display:flex;justify-content:space-between;gap:14px;padding:6px 0;border-bottom:1px solid #191919}
 .srow:last-child{border-bottom:none}
 .srow .k{color:#888} .srow .v{color:#eee;font-family:ui-monospace,monospace;text-align:right;word-break:break-word}
 .badge{padding:2px 9px;border-radius:10px;font-size:12px}
 .badge.run{background:#1f6f3f;color:#9be8b4} .badge.pause{background:#7a5b1e;color:#f0d59a}
 .badge.done{background:#1f5fae;color:#bcd9ff} .badge.load{background:#333;color:#bbb}
 .badge.stop{background:#5a2d2d;color:#f3b0b0}
</style></head><body><div class="wrap">
<h1>Interactive LIBERO · π0.5</h1>
<div class="row">
 <img id="view" src="/frame.jpg">
 <div class="panel">
  <label>Task &mdash; selecting one loads its scene (objects), paused &amp; ready</label>
  <div class="row" style="gap:8px">
   <select id="suite" style="flex:1">
    <option>libero_10</option><option>libero_goal</option>
    <option>libero_object</option><option>libero_spatial</option><option>libero_90</option>
   </select>
   <select id="task" style="flex:2"></select>
  </div>

  <label>Instruction to π0.5 &mdash; blank uses the task's own goal</label>
  <div class="row" style="gap:8px">
   <input type="text" id="instr" style="flex:1" placeholder="e.g. pick up the bowl">
   <button id="send" class="green">Send</button>
  </div>
  <div class="hint" id="canon"></div>

  <div class="row" style="gap:8px;margin-top:16px">
   <button id="play" class="big" style="flex:2">▶ Play</button>
   <button id="reset" class="alt big" style="flex:1">↻ Reset</button>
  </div>
  <div class="hint">Send a new instruction any time &mdash; corrections or staged subgoals. Objects must exist in the loaded scene.</div>

  <div class="row" style="gap:8px;margin-top:14px;align-items:center">
   <button id="savevid" class="alt" style="flex:2">💾 Save video</button>
   <label style="margin:0;color:#aaa">speed</label>
   <select id="speed" style="flex:0 0 auto">
    <option value="1">1×</option><option value="2" selected>2×</option>
    <option value="4">4×</option><option value="8">8×</option>
   </select>
  </div>
  <div class="hint">Saves the run-so-far with the active prompt captioned below each frame; downloads to your laptop and keeps a copy in <code>runs/videos/</code> on the pod.</div>

  <div class="toast" id="toast"></div>
  <div class="status" id="status"></div>
 </div>
</div></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
// Primary: MJPEG stream (one connection, bandwidth-bound, smooth even over a
// distant/proxied link). Fallback: if the stream fails to start within a couple
// seconds (e.g. a proxy that buffers multipart), switch to onload-chained polling
// of /frame.jpg (each fetch waits for the previous to finish, so no aborted loads).
const view=$('view');
let streaming=false;
function pollFrame(){ view.src='/frame.jpg?t='+Date.now(); }
function startPolling(){ view.onload=()=>setTimeout(pollFrame,20); view.onerror=()=>setTimeout(pollFrame,150); pollFrame(); }
function startStream(){
 view.onload=()=>{ streaming=true; };
 view.onerror=()=>{ if(!streaming) startPolling(); };
 view.src='/stream.mjpg';
 setTimeout(()=>{ if(!streaming) startPolling(); }, 2500);
}
startStream();

let toastTimer;
function toast(msg,color){ const t=$('toast'); t.textContent=msg; t.style.color=color||'#3fb950'; t.classList.add('show');
 clearTimeout(toastTimer); toastTimer=setTimeout(()=>t.classList.remove('show'),2600); }

async function loadTasks(){
 const s=$('suite').value;
 const r=await fetch('/tasks?suite='+s); const j=await r.json();
 $('task').innerHTML=j.tasks.map(t=>`<option value="${t.task_id}">Task ${t.task_id} — ${t.language}</option>`).join('');
 showCanon();
}
function canonGoal(){ const o=$('task').selectedOptions[0]; return o?o.textContent.replace(/^Task \\d+ — /,''):''; }
function showCanon(){ const g=canonGoal(); $('canon').textContent=g?('default goal: '+g):''; }

// Selecting a task (or suite) auto-loads its scene, paused & ready for Play.
async function doLoad(){
 const instr=$('instr').value.trim();
 toast('Loading scene… (paused — press Play to start)','#d8a657');
 await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({suite:$('suite').value,task_id:parseInt($('task').value),init_id:0,instruction:instr})});
}
$('suite').onchange=async()=>{ await loadTasks(); doLoad(); };
$('task').onchange=doLoad;
$('reset').onclick=()=>{ toast('Reset — press Play to start','#d8a657'); doLoad(); };

$('send').onclick=async()=>{
 const t=$('instr').value.trim();
 if(!t){ toast('Type an instruction first.','#e06c75'); return; }
 await fetch('/instruction',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
 toast(serverPaused?'Instruction set ✓ — press Play':'Instruction sent ✓');
};
$('instr').addEventListener('keydown',e=>{if(e.key==='Enter')$('send').click();});

$('savevid').onclick=async()=>{
 const def='task'+($('task').value||'0')+'_'+new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
 const name=prompt('Name this video:', def);
 if(name===null) return;                       // cancelled
 const speed=$('speed').value;
 const btn=$('savevid'), orig=btn.textContent; btn.disabled=true; btn.textContent='Rendering…';
 toast('Rendering video ('+speed+'×)…','#d8a657');
 try{
  const r=await fetch('/save_video',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,speed})});
  if(!r.ok){ let j={}; try{j=await r.json();}catch(e){} toast(j.error||'Save failed','#e06c75'); return; }
  const blob=await r.blob(), url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=(name||'rollout')+'.mp4';
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  toast('Saved ✓ — downloaded + kept in runs/videos/ on the pod');
 }catch(e){ toast('Save failed: '+e,'#e06c75'); }
 finally{ btn.disabled=false; btn.textContent=orig; }
};

let serverPaused=true, limitReached=false;
function syncPlay(p){ serverPaused=p; const b=$('play'); b.textContent=p?'▶ Play':'⏸ Pause'; b.classList.toggle('ready',p); }
$('play').onclick=async()=>{
 if(limitReached){ toast('Reached the step limit — press Reset to run again.','#e0a23b'); return; }
 const np=!serverPaused; syncPlay(np);
 await fetch('/pause',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused:np})});
 toast(np?'Paused':'Playing…','#d8a657');
};

function row(k,v){ return `<div class="srow"><span class="k">${k}</span><span class="v">${v}</span></div>`; }
async function poll(){
 try{ const r=await fetch('/status'); const s=await r.json();
  syncPlay(s.paused); limitReached=s.limit_reached;
  // Solving no longer ends the episode: 'solved' is a non-terminal indicator.
  const badge = !s.connected ? '<span class="badge load">loading…</span>'
    : s.limit_reached ? `<span class="badge stop">⏹ step limit (${s.step_limit}) — Reset</span>`
    : s.paused ? '<span class="badge pause">⏸ paused</span>'
    : '<span class="badge run">▶ running</span>';
  const solved = s.success ? ' <span class="badge done">✓ goal met</span>' : '';
  // "Sent to policy" = ground truth captured inside the inference call.
  const sent = s.sent_prompt
    ? `${esc(s.sent_prompt)} <span class="k">(@step ${s.sent_step})</span>`
    : '<span class="k">— nothing sent yet (press Play) —</span>';
  const pending = (s.instruction && s.instruction !== s.sent_prompt)
    ? ' <span class="badge pause">pending — not executed yet</span>' : '';
  $('status').innerHTML =
    row('Task', `${s.task_id} · ${esc(s.suite)}`)
   +row('Scene goal', esc(s.task_language)||'—')
   +row('Prompt (set)', (esc(s.instruction)||'—')+pending)
   +row('→ Sent to policy', sent)
   +row('Step', `${s.step} / ${s.step_limit}`)
   +row('State', badge+solved);
 }catch(e){}
 setTimeout(poll,400);
}
(async()=>{ await loadTasks(); doLoad(); poll(); })();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description="Interactive LIBERO runner for pi0.5")
    p.add_argument("--host", default="0.0.0.0", help="policy server host")
    p.add_argument("--port", type=int, default=8000, help="policy server port")
    p.add_argument("--web-port", type=int, default=8888, help="web UI port")
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--max-rollout-steps", type=int, default=5000,
                   help="auto-stop a rollout after this many steps (Reset to run again)")
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()

    worker = RolloutWorker(args)
    worker.start()
    app = build_app(worker, args)
    logger.info("Web UI on :%d  (policy server %s:%d)", args.web_port, args.host, args.port)
    app.run(host="0.0.0.0", port=args.web_port, threaded=True)


if __name__ == "__main__":
    main()
