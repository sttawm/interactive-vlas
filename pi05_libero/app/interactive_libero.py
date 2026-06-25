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
import threading
import time

# MuJoCo must render offscreen on a headless GPU box. EGL uses the GPU; override
# with MUJOCO_GL=osmesa for CPU rendering if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

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
        self._reset_instruction = ""  # custom prompt to start the next rollout with
        self._clear_plan = False  # force a replan now (e.g. after an instruction change)

        # Status (guarded by _lock).
        self.status = {
            "suite": args.task_suite_name,
            "task_id": args.task_id,
            "task_language": "",
            "instruction": "",
            "step": 0,
            "max_steps": MAX_STEPS_BY_SUITE.get(args.task_suite_name, 300),
            "paused": True,
            "done": False,
            "success": False,
            "connected": False,
        }

        # Per-rollout recording (only touched by worker thread).
        self._record_frames = []
        self._record_actions = []
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
        with self._lock:
            self._reset_to = (suite, int(task_id), int(init_id))
            self._reset_instruction = (instruction or "").strip()
            self._paused = True
            self.status["paused"] = True

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
                with self._lock:
                    reset_instruction = self._reset_instruction
                    self._reset_instruction = ""
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
                # Use the typed custom prompt if given; otherwise the canonical task language.
                with self._lock:
                    self._instruction = reset_instruction or str(task_language)
                    self.status.update(
                        suite=suite,
                        task_id=task_id,
                        task_language=str(task_language),
                        instruction=self._instruction,
                        step=0,
                        max_steps=MAX_STEPS_BY_SUITE.get(suite, 300),
                        done=False,
                        success=False,
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
                done = self.status["done"]
            if done:
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
            action_chunk = self.client.infer(element)["actions"]
            action_plan.extend(action_chunk[: self.args.replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        self._record_frames.append(img)
        self._record_actions.append(np.asarray(action))
        self._publish_frame(obs, overlay=instruction, step=step + 1)

        with self._lock:
            self.status["step"] = step + 1
            if done:
                self.status["done"] = True
                self.status["success"] = True
        if done:
            logger.info("Task solved at step %s", step + 1)
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

let serverPaused=true;
function syncPlay(p){ serverPaused=p; const b=$('play'); b.textContent=p?'▶ Play':'⏸ Pause'; b.classList.toggle('ready',p); }
$('play').onclick=async()=>{ const np=!serverPaused; syncPlay(np);
 await fetch('/pause',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused:np})});
 toast(np?'Paused':'Playing…','#d8a657');
};

function row(k,v){ return `<div class="srow"><span class="k">${k}</span><span class="v">${v}</span></div>`; }
async function poll(){
 try{ const r=await fetch('/status'); const s=await r.json();
  syncPlay(s.paused);
  const badge = !s.connected ? '<span class="badge load">loading…</span>'
    : s.success ? '<span class="badge done">solved ✓</span>'
    : s.paused ? '<span class="badge pause">⏸ paused</span>'
    : '<span class="badge run">▶ running</span>';
  $('status').innerHTML =
    row('Task', `${s.task_id} · ${esc(s.suite)}`)
   +row('Scene goal', esc(s.task_language)||'—')
   +row('Prompt', esc(s.instruction)||'—')
   +row('Step', `${s.step}`)
   +row('State', badge);
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
