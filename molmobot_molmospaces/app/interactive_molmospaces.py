"""Interactive MolmoSpaces runner for MolmoBot.

Build a MolmoSpaces MuJoCo scene (Franka DROID in a ProcTHOR house), type a
language instruction in the browser, and watch the MolmoBot VLA act on it in real
time. Change the instruction mid-rollout (corrections / staged subgoals).

This deliberately reuses MolmoBot's own working in-process demo loop
(`MolmoBot/demo_policy.ipynb`): we build the scene with `MjSpec`, render the two
policy cameras with `mujoco.Renderer`, and drive a single `RealRobotVLAPolicy`
directly — no websocket, no parallel datagen runner. The only real change is that
the prompt (`obs["task"]`) is a live, user-controlled variable.

Two policy backends, selected at launch:
  - real  (default): MolmoBot `RealRobotVLAPolicy` loaded from a checkpoint. Needs a
    CUDA GPU (the ~4B flow-matching model). Run this on the pod.
  - stub  (--stub): no model — a scripted joint wobble. Lets you validate the whole
    rig (scene build, rendering, browser stream, instruction plumbing, run logging)
    locally on a Mac/CPU with no GPU and without installing MolmoBot's `olmo` package.

The scene build needs only `molmo_spaces` (+ mujoco); the real policy additionally
needs MolmoBot's `olmo` package, which is imported lazily so --stub works without it.
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import json
import logging
import os
import pathlib
import sys
import threading
import time

# MuJoCo must render offscreen. On a headless Linux pod use EGL (the GPU); on macOS
# the default (CGL) is correct, so only force EGL on Linux — matching the MolmoBot demo.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

import mujoco
from mujoco import MjData, MjModel, MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses, get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
    install_uid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("interactive_molmospaces")

# The MolmoBot demo runs the policy at ~15 Hz; nstep is derived from the model timestep.
DEFAULT_POLICY_DT_MS = 66
CAM_EXO = "robot_0/exo_camera_1"
CAM_WRIST = "robot_0/gripper/wrist_camera"


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
class StubPolicy:
    """No-model stand-in: a gentle, visible joint wobble.

    Validates the sim/render/stream/logging rig without a GPU or the MolmoBot model.
    It ignores the typed instruction (a scripted policy can't be steered by language);
    use it only to confirm the plumbing, then switch to the real policy on a GPU pod.
    """

    name = "stub"

    def __init__(self, view):
        self._view = view
        self._t = 0

    def reset(self):
        self._t = 0

    def get_action(self, obs):
        self._t += 1
        arm = self._view.get_move_group("arm")
        grip = self._view.get_move_group("gripper")
        arm_ctrl = np.asarray(arm.noop_ctrl, dtype=float).copy()
        amp = 0.3
        arm_ctrl[0] += amp * np.sin(self._t * 0.08)
        if arm_ctrl.shape[0] > 3:
            arm_ctrl[3] += amp * np.sin(self._t * 0.08 + 1.0)
        return {"arm": arm_ctrl, "gripper": np.asarray(grip.noop_ctrl, dtype=float)}


def load_real_policy(checkpoint, action_type):
    """Instantiate MolmoBot's RealRobotVLAPolicy (lazy import; needs `olmo` + a GPU).

    Mirrors MolmoBot/demo_policy.ipynb cell 7.
    """
    from huggingface_hub import snapshot_download
    from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig

    ckpt_path = checkpoint
    if not os.path.isdir(checkpoint):
        logger.info("Resolving checkpoint %s via snapshot_download...", checkpoint)
        ckpt_path = snapshot_download(checkpoint)
    logger.info("Checkpoint at %s", ckpt_path)

    class _MockConfig:
        def __init__(self, policy_config):
            self.policy_config = policy_config

    policy_config = RealRobotVLAPolicyConfig()
    policy_config.checkpoint_path = ckpt_path
    policy_config.action_type = action_type           # MolmoBot-DROID = absolute "joint_pos"
    policy_config.action_keys["arm"] = action_type
    policy = RealRobotVLAPolicy(config=_MockConfig(policy_config), task_type="manipulation")
    policy.name = "molmobot"
    return policy


# --------------------------------------------------------------------------- #
# Sim world (built inside the worker thread — MuJoCo contexts are thread-affine)
# --------------------------------------------------------------------------- #
class SimWorld:
    """A compiled MolmoSpaces scene + Franka DROID + the two policy cameras.

    Faithfully reproduces MolmoBot/demo_policy.ipynb cell 3.
    """

    def __init__(self, house_index, split, render_h=360, render_w=640, add_bowl=True):
        houses = get_procthor_10k_houses(split=split)
        entries = houses[split]
        house_index = max(0, min(house_index, len(entries) - 1))
        house_xml_path = entries[house_index]["base"]
        install_scene_with_objects_and_grasps_from_path(house_xml_path)

        spec = MjSpec.from_file(house_xml_path)

        robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
        # add_robot_to_scene drifted between molmo_spaces versions: MolmoBot pins an
        # older one that takes a positional `robot_spec`; newer ones load it internally.
        # Support both so this runs against the pod's pinned version AND a fresh clone.
        add_kwargs = dict(
            prefix=robot_config.robot_namespace,
            pos=[6.8, 9.75],
            quat=R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True),
        )
        if "robot_spec" in inspect.signature(FrankaRobot.add_robot_to_scene).parameters:
            robot_file_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
            robot_spec = MjSpec.from_file(str(robot_file_path))
            FrankaRobot.add_robot_to_scene(robot_config, spec, robot_spec, **add_kwargs)
        else:
            FrankaRobot.add_robot_to_scene(robot_config, spec, **add_kwargs)
        FrankaRobot.apply_control_overrides(spec, robot_config)

        ns = robot_config.robot_namespace
        spec.camera(ns + "gripper/wrist_camera").resolution = [render_w, render_h]
        spec.body(ns + "fr3_link0").add_camera(
            pos=[0.1, 0.57, 0.66],
            quat=[-0.3633, -0.1241, 0.4263, 0.8191],
            fovy=71.0,
            resolution=[render_w, render_h],
            name="robot_0/exo_camera_1",
        )

        # The demo attaches a bowl receptacle at house-0's counter; only meaningful there.
        if add_bowl and house_index == 0:
            try:
                bowl_spec = MjSpec.from_file(str(install_uid("Bowl_3")))
                frame = spec.worldbody.add_frame(
                    pos=[7.1, 10.2, 1.01],
                    quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True),
                )
                frame.attach_body(bowl_spec.worldbody.first_body(), prefix="place_receptacle/")
            except Exception:
                logger.exception("Could not attach demo bowl receptacle (continuing without it)")

        self.model: MjModel = spec.compile()
        self.data = MjData(self.model)
        self.view = FrankaDroidRobotView(self.data, ns)
        self.robot_config = robot_config
        self.house_index = house_index

        self.view.set_qpos_dict(robot_config.init_qpos)
        mujoco.mj_forward(self.model, self.data)
        for mg_id in self.view.move_group_ids():
            mg = self.view.get_move_group(mg_id)
            mg.ctrl = mg.noop_ctrl
        mujoco.mj_forward(self.model, self.data)

        self._renderer = mujoco.Renderer(self.model, render_h, render_w)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.sitegroup = 0
        # nstep per policy step (e.g. 66ms / 2ms = 33), >=1.
        self.nstep = max(1, DEFAULT_POLICY_DT_MS // max(1, round(self.model.opt.timestep * 1000)))

    def render(self):
        self._renderer.update_scene(self.data, camera=CAM_EXO, scene_option=self._scene_option)
        exo = self._renderer.render()
        self._renderer.update_scene(self.data, camera=CAM_WRIST, scene_option=self._scene_option)
        wrist = self._renderer.render()
        return {"exo_camera_1": exo, "wrist_camera": wrist}

    def obs(self, instruction):
        return {
            "task": instruction,
            "qpos": {
                "arm": self.view.get_move_group("arm").joint_pos,
                "gripper": self.view.get_move_group("gripper").joint_pos,
            },
            **self.render(),
        }

    def apply(self, action):
        for mg_id in action:
            self.view.get_move_group(mg_id).ctrl = action[mg_id]
        mujoco.mj_step(self.model, self.data, nstep=self.nstep)


# --------------------------------------------------------------------------- #
# Rollout worker
# --------------------------------------------------------------------------- #
class RolloutWorker(threading.Thread):
    def __init__(self, args, policy_factory):
        super().__init__(daemon=True)
        self.args = args
        self._policy_factory = policy_factory  # (view) -> policy ; called after scene build
        self._policy = None

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._instruction = args.instruction or ""
        self._latest_jpeg = _placeholder_jpeg("starting... load a scene")
        self._paused = True
        self._reset_to = args.house_index  # request an initial build
        self._force_refresh = False

        self.status = {
            "house_index": args.house_index,
            "split": args.split,
            "instruction": self._instruction,
            "step": 0,
            "max_steps": args.max_steps,
            "paused": True,
            "policy": "(loading)",
            "ready": False,
        }

        self._record_frames = []
        self._record_actions = []
        self._record_instructions = []
        self._run_dir = None
        self._run_meta = {}

    # ----- public API used by Flask handlers -----
    def set_instruction(self, text):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._instruction = text
            self._force_refresh = True  # re-query the policy now instead of waiting for the buffer
            self.status["instruction"] = text
            step = self.status["step"]
        self._record_instructions.append(
            [datetime.datetime.now().isoformat(timespec="seconds"), step, text]
        )
        logger.info("New instruction @step %s: %r", step, text)

    def request_reset(self, house_index):
        with self._lock:
            self._reset_to = int(house_index)
            self._paused = False
            self.status["paused"] = False

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
    def _start_new_run(self, world):
        self._finalize_run()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = pathlib.Path(self.args.runs_dir) / f"{stamp}_house{world.house_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._record_frames = []
        self._record_actions = []
        self._record_instructions = []
        self._run_meta = {
            "house_index": world.house_index,
            "split": self.args.split,
            "policy": self._policy.name,
            "checkpoint": self.args.checkpoint,
            "action_type": self.args.action_type,
            "policy_dt_ms": DEFAULT_POLICY_DT_MS,
            "seed": self.args.seed,
            "started": stamp,
        }
        logger.info("Recording run to %s", run_dir)

    def _finalize_run(self):
        if not self._run_dir:
            return
        try:
            if self._record_frames:
                imageio.mimwrite(str(self._run_dir / "rollout.mp4"), self._record_frames, fps=15)
            if self._record_actions:
                np.save(str(self._run_dir / "actions.npy"),
                        np.asarray(self._record_actions, dtype=object), allow_pickle=True)
            with open(self._run_dir / "instructions.txt", "w") as fh:
                for wall, step, text in self._record_instructions:
                    fh.write(f"{wall}\tstep={step}\t{text}\n")
            meta = dict(self._run_meta)
            with self._lock:
                meta["final_step"] = self.status["step"]
            meta["ended"] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            with open(self._run_dir / "metadata.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            logger.info("Saved run: %s (%d frames)", self._run_dir, len(self._record_frames))
        except Exception:
            logger.exception("Failed to finalize run")
        finally:
            self._run_dir = None

    def run(self):
        world = None
        step = 0
        while not self._stop.is_set():
            with self._lock:
                reset_to = self._reset_to
                self._reset_to = None
                paused = self._paused
                instruction = self._instruction
                force_refresh = self._force_refresh
                self._force_refresh = False

            if reset_to is not None:
                self._finalize_run()
                logger.info("Building scene for house %s (%s split)...", reset_to, self.args.split)
                with self._lock:
                    self.status.update(ready=False, step=0)
                try:
                    world = SimWorld(reset_to, self.args.split, add_bowl=self.args.add_bowl)
                except Exception:
                    logger.exception("Scene build failed")
                    self._latest_jpeg = _placeholder_jpeg("scene build failed (see log)")
                    world = None
                    self.set_paused(True)
                    continue
                if self._policy is None:
                    logger.info("Loading policy...")
                    self._policy = self._policy_factory(world.view)
                    with self._lock:
                        self.status["policy"] = self._policy.name
                elif isinstance(self._policy, StubPolicy):
                    self._policy._view = world.view  # rebind stub to the new scene
                if hasattr(self._policy, "reset"):
                    try:
                        self._policy.reset()
                    except Exception:
                        logger.exception("policy.reset() failed (continuing)")
                step = 0
                self._start_new_run(world)
                with self._lock:
                    if not self._instruction:
                        self._instruction = "pick up an object"
                    self.status.update(
                        house_index=world.house_index, instruction=self._instruction,
                        step=0, ready=True,
                    )
                self._record_instructions.append(
                    [datetime.datetime.now().isoformat(timespec="seconds"), 0, self._instruction]
                )
                self._publish_frame(world, self._instruction, 0)
                continue

            if world is None or paused or step >= self.args.max_steps:
                if world is not None and step >= self.args.max_steps:
                    with self._lock:
                        self.status["paused"] = True
                    self._paused = True
                time.sleep(0.05)
                continue

            try:
                if force_refresh:
                    self._do_force_refresh()
                obs = world.obs(instruction)
                action = self._policy.get_action(obs)
                self._record_frames.append(_stack(obs))
                self._record_actions.append({k: np.asarray(v) for k, v in action.items()})
                world.apply(action)
                step += 1
                self._publish_frame(world, instruction, step)
                with self._lock:
                    self.status["step"] = step
            except Exception:
                logger.exception("Step failed; pausing rollout")
                self.set_paused(True)

        self._finalize_run()

    def _do_force_refresh(self):
        """Make a mid-rollout instruction change take effect immediately.

        RealRobotVLAPolicy only re-reads obs["task"] when its action buffer refills
        (every execute_horizon steps). Emptying the buffer forces a re-query now.
        """
        p = self._policy
        try:
            if hasattr(p, "action_buffer"):
                p.action_buffer = []
            if hasattr(p, "buffer_index") and hasattr(p, "execute_horizon"):
                p.buffer_index = p.execute_horizon
        except Exception:
            logger.exception("force-refresh failed (instruction will apply at next buffer refill)")

    def _publish_frame(self, world, overlay, step):
        frame = _stack({"exo_camera_1": None})  # placeholder if render fails
        try:
            imgs = world.render()
            frame = np.hstack([imgs["exo_camera_1"], imgs["wrist_camera"]])
        except Exception:
            logger.exception("render failed")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h = frame.shape[0]
        if overlay:
            label = overlay if len(overlay) < 80 else overlay[:77] + "..."
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(frame, f"{label}   [step {step}]", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()


def _stack(obs):
    if obs.get("exo_camera_1") is None:
        return np.zeros((360, 1280, 3), dtype=np.uint8)
    return np.hstack([obs["exo_camera_1"], obs["wrist_camera"]])


def _placeholder_jpeg(text):
    img = np.zeros((360, 1280, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
def build_app(worker, args):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/frame.jpg")
    def frame():
        return Response(worker.latest_jpeg(), mimetype="image/jpeg")

    @app.route("/stream.mjpg")
    def stream():
        def gen():
            last = None
            while True:
                jpg = worker.latest_jpeg()
                if jpg is not last:
                    last = jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.04)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return jsonify(worker.snapshot_status())

    @app.route("/instruction", methods=["POST"])
    def instruction():
        worker.set_instruction((request.json or {}).get("text", ""))
        return jsonify(ok=True)

    @app.route("/reset", methods=["POST"])
    def reset():
        body = request.json or {}
        worker.request_reset(body.get("house_index", args.house_index))
        return jsonify(ok=True)

    @app.route("/pause", methods=["POST"])
    def pause():
        worker.set_paused((request.json or {}).get("paused", True))
        return jsonify(ok=True)

    return app


INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Interactive MolmoSpaces · MolmoBot</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
 .wrap{max-width:1180px;margin:0 auto}
 h1{font-size:18px;font-weight:600}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 #view{width:1024px;max-width:100%;background:#000;border:1px solid #333;border-radius:8px}
 .panel{flex:1;min-width:300px}
 input,select,button,textarea{font-size:14px;padding:8px;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#eee}
 input[type=text],input[type=number]{width:100%}
 button{background:#2d6cdf;border:none;cursor:pointer}
 button.alt{background:#444}
 .status{font-family:monospace;font-size:13px;background:#000;padding:10px;border-radius:6px;white-space:pre-wrap;line-height:1.5}
 label{font-size:12px;color:#aaa;display:block;margin:10px 0 4px}
 .hint{color:#888;font-size:12px;margin-top:4px}
</style></head><body><div class="wrap">
<h1>Interactive MolmoSpaces · MolmoBot <span style="color:#888;font-weight:400">(left: exo cam · right: wrist cam)</span></h1>
<img id="view" src="/frame.jpg">
<div class="row" style="margin-top:14px">
 <div class="panel">
  <label>Scene (ProcTHOR-10k house index)</label>
  <div class="row" style="gap:8px">
   <input type="number" id="house" value="0" min="0" style="flex:1">
   <button id="load" style="flex:2">Load scene &amp; start</button>
  </div>
  <div class="hint">House 0 is the demo kitchen (with a bowl). Other indices load different houses.</div>

  <label>Instruction (type anything; objects must exist in the scene)</label>
  <input type="text" id="instr" placeholder="e.g. put the salt shaker in the bowl">
  <div class="row" style="gap:8px;margin-top:8px">
   <button id="send" style="flex:2">Send instruction</button>
   <button id="pause" class="alt" style="flex:1">Pause</button>
   <button id="reset" class="alt" style="flex:1">Reset</button>
  </div>
  <div class="hint">A new instruction forces an immediate replan — use it for corrections or staged subgoals.</div>
 </div>
 <div class="panel">
  <label>Status</label>
  <div class="status" id="status">idle</div>
 </div>
</div></div>
<script>
const $=id=>document.getElementById(id);
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
$('load').onclick=async()=>{
 await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({house_index:parseInt($('house').value)||0})});
};
$('send').onclick=async()=>{
 await fetch('/instruction',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({text:$('instr').value})});
};
$('instr').addEventListener('keydown',e=>{if(e.key==='Enter')$('send').click();});
let paused=false;
$('pause').onclick=async()=>{ paused=!paused;
 await fetch('/pause',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused})});
 $('pause').textContent=paused?'Resume':'Pause';
};
$('reset').onclick=()=>$('load').click();
async function poll(){
 try{ const r=await fetch('/status'); const s=await r.json();
  $('status').textContent=
   `policy:   ${s.policy}\nhouse:    ${s.house_index}  (${s.split})\n`+
   `prompt:   ${s.instruction}\nstep:     ${s.step} / ${s.max_steps}\n`+
   `paused:   ${s.paused}\nready:    ${s.ready}`;
 }catch(e){}
 setTimeout(poll,400);
}
poll();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description="Interactive MolmoSpaces runner for MolmoBot")
    p.add_argument("--web-port", type=int, default=8888, help="web UI port")
    p.add_argument("--house-index", type=int, default=0, help="ProcTHOR-10k house index")
    p.add_argument("--split", default="val", help="house split (val/train/test)")
    p.add_argument("--instruction", default="", help="initial instruction")
    p.add_argument("--max-steps", type=int, default=600, help="policy steps before auto-pause")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--no-bowl", dest="add_bowl", action="store_false",
                   help="don't attach the demo bowl receptacle on house 0")
    # Policy
    p.add_argument("--stub", action="store_true",
                   help="use the no-model wobble policy (local plumbing test, no GPU)")
    p.add_argument("--checkpoint", default="allenai/MolmoBot-DROID",
                   help="HF repo id or local dir for the real policy")
    p.add_argument("--action-type", default="joint_pos",
                   help="MolmoBot-DROID uses absolute joint_pos")
    args = p.parse_args()

    if args.stub:
        policy_factory = lambda view: StubPolicy(view)  # noqa: E731
        logger.info("Using STUB policy (no model). Instruction will NOT steer the robot.")
    else:
        policy_factory = lambda view: load_real_policy(args.checkpoint, args.action_type)  # noqa: E731

    worker = RolloutWorker(args, policy_factory)
    worker.start()
    app = build_app(worker, args)
    logger.info("Web UI on :%d", args.web_port)
    app.run(host="0.0.0.0", port=args.web_port, threaded=True)


if __name__ == "__main__":
    main()
