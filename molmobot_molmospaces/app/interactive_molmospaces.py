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
import re
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

# Shared web UI (one page reused by every VLA+env instance). Repo root is three
# levels up: molmobot_molmospaces/app/<this file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from shared.webui import build_app  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("interactive_molmospaces")

# The MolmoBot demo runs the policy at ~15 Hz; nstep is derived from the model timestep.
DEFAULT_POLICY_DT_MS = 66
CAM_EXO = "robot_0/exo_camera_1"
CAM_WRIST = "robot_0/gripper/wrist_camera"
ENV_NAME = "MolmoSpaces (ProcTHOR)"


def molmobot_available():
    import importlib.util
    return importlib.util.find_spec("olmo") is not None


def _house_from_scene(scene, default):
    """Parse a scene label like 'house 3' -> 3."""
    if not scene:
        return default
    try:
        return int(str(scene).strip().split()[-1])
    except (ValueError, IndexError):
        return default


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
    def __init__(self, args, available_vlas):
        super().__init__(daemon=True)
        self.args = args
        self.available_vlas = available_vlas
        self._default_vla = available_vlas[0]
        self._real_policy = None  # cached MolmoBot policy (loaded once, reused)
        self._policy = None
        self._cur_vla = self._default_vla

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._instruction = args.instruction or self._scene_default_prompt(args.house_index)
        self._latest_jpeg = _placeholder_jpeg("pick a VLA + scene")
        self._paused = True
        self._reset_to = {"vla": self._default_vla, "house_index": args.house_index}
        self._force_refresh = False

        self.status = {
            "vla": self._default_vla,
            "env": ENV_NAME,
            "house": args.house_index,
            "instruction": self._instruction,
            "sent_prompt": "",
            "sent_step": 0,
            "step": 0,
            "max_steps": args.max_steps,
            "paused": True,
            "limit_reached": False,
            "ready": False,
        }

        self._record_frames = []
        self._record_actions = []
        self._record_prompts = []  # prompt active at each recorded frame (for captions)
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

    def _scene_default_prompt(self, house_index):
        # House 0 carries the demo bowl + knife (the verified default); others generic.
        return "pick up the knife" if house_index == 0 else "pick up an object"

    def config(self):
        scenes = [{"value": f"house {i}", "label": f"house {i}",
                   "default_prompt": self._scene_default_prompt(i)}
                  for i in range(self.args.num_scenes)]
        return {
            "title": "MolmoBot · MolmoSpaces",
            "selectors": [
                {"name": "vla", "label": "VLA", "options": self.available_vlas},
                {"name": "env", "label": "Env", "options": [ENV_NAME]},
                {"name": "scene", "label": "Scene", "depends_on": "env",
                 "options_by": {ENV_NAME: scenes}},
            ],
            "instruction_label": "Instruction to MolmoBot — blank uses the scene's default",
            "instruction_placeholder": "e.g. pick up the knife",
        }

    def default_prompt(self, selection):
        house = _house_from_scene(selection.get("scene"), self.args.house_index)
        return self._scene_default_prompt(house)

    def request_reset(self, selection, instruction=""):
        vla = selection.get("vla") or self._cur_vla
        if vla not in self.available_vlas:
            vla = self._default_vla
        house = _house_from_scene(selection.get("scene"), self.args.house_index)
        # Resolve the default prompt synchronously and set the prompt now, so a Send
        # during the (~minute) scene/model load isn't clobbered. Land paused & ready.
        instruction = (instruction or "").strip()
        with self._lock:
            self._reset_to = {"vla": vla, "house_index": house}
            self._instruction = instruction or self._scene_default_prompt(house)
            self._paused = True
            self.status["paused"] = True
            self.status["instruction"] = self._instruction
            # Clear prior-run state synchronously so the UI never flashes "step limit
            # reached" while the new scene builds.
            self.status["limit_reached"] = False
            self.status["step"] = 0
            self.status["ready"] = False

    def _make_policy(self, vla, view):
        if vla == "stub":
            return StubPolicy(view)
        if self._real_policy is None:
            logger.info("Loading MolmoBot policy (first load may take a minute)...")
            self._real_policy = load_real_policy(self.args.checkpoint, self.args.action_type)
        return self._real_policy

    def set_paused(self, paused):
        with self._lock:
            self._paused = bool(paused)
            self.status["paused"] = bool(paused)

    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def snapshot_status(self):
        with self._lock:
            s = dict(self.status)
        if not s["ready"]:
            state = "loading…"
        elif s["limit_reached"]:
            state = "⏹ step limit (%d) — Reset" % s["max_steps"]
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
            # VLA / Env / Scene are already shown in the selectors above — don't duplicate.
            "Prompt (set)": prompt,
            "→ Sent to policy": sent,
            "Step": "%d / %d" % (s["step"], s["max_steps"]),
            "State": state,
            # consumed by the frontend (Play button + step-limit handling), not displayed:
            "paused": s["paused"],
            "limit_reached": s["limit_reached"],
        }

    def save_video(self, name, speed=1.0):
        with self._lock:
            frames = list(self._record_frames)
            prompts = list(self._record_prompts)
        return _compose_wide_video(frames, prompts, name, speed, self.args.runs_dir)

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
        self._record_prompts = []
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
                vla = reset_to["vla"]
                house_index = reset_to["house_index"]
                self._finalize_run()
                logger.info("Building house %s (%s) for VLA=%s ...", house_index, self.args.split, vla)
                with self._lock:
                    self.status.update(ready=False, step=0, vla=vla, house=house_index)
                try:
                    world = SimWorld(house_index, self.args.split, add_bowl=self.args.add_bowl)
                except Exception:
                    logger.exception("Scene build failed")
                    self._latest_jpeg = _placeholder_jpeg("scene build failed (see log)")
                    world = None
                    self.set_paused(True)
                    continue
                try:
                    self._policy = self._make_policy(vla, world.view)
                except Exception:
                    logger.exception("Policy load failed")
                    self._latest_jpeg = _placeholder_jpeg(f"failed to load VLA '{vla}' (see log)")
                    self.set_paused(True)
                    continue
                self._cur_vla = vla
                if hasattr(self._policy, "reset"):
                    try:
                        self._policy.reset()
                    except Exception:
                        logger.exception("policy.reset() failed (continuing)")
                step = 0
                self._start_new_run(world)
                # The prompt was set synchronously in request_reset (so a Send during
                # this slow load isn't clobbered) — don't reassign it here.
                with self._lock:
                    self.status.update(
                        vla=vla, house=world.house_index, instruction=self._instruction,
                        step=0, sent_prompt="", sent_step=0, limit_reached=False, ready=True,
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
                        self.status["limit_reached"] = True
                    self._paused = True
                time.sleep(0.05)
                continue

            try:
                if force_refresh:
                    self._do_force_refresh()
                obs = world.obs(instruction)
                with self._lock:
                    self.status["sent_prompt"] = str(instruction)  # ground truth: obs["task"]
                    self.status["sent_step"] = step + 1
                action = self._policy.get_action(obs)
                self._record_frames.append(_stack(obs))
                self._record_actions.append({k: np.asarray(v) for k, v in action.items()})
                self._record_prompts.append(str(instruction))
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
            # Stack the two cameras VERTICALLY (exo over wrist) -> near-square ~640x720,
            # which displays well in the shared square-ish view (side-by-side was a thin
            # squished strip).
            frame = np.vstack([imgs["exo_camera_1"], imgs["wrist_camera"]])
        except Exception:
            logger.exception("render failed")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if overlay:
            label = overlay if len(overlay) < 58 else overlay[:55] + "..."
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(frame, f"{label}   [step {step}]", (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()


def _stack(obs):
    if obs.get("exo_camera_1") is None:
        return np.zeros((720, 640, 3), dtype=np.uint8)
    return np.vstack([obs["exo_camera_1"], obs["wrist_camera"]])


def _placeholder_jpeg(text):
    img = np.zeros((720, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _compose_wide_video(frames, prompts, name, speed=1.0, runs_dir="runs"):
    """Captioned, speed-adjustable export of the dual-cam (wide) rollout. Keeps the
    native frame aspect (shared.webui.compose_video assumes square) and captions the
    active prompt below each frame. Returns the mp4 path, or None if no frames."""
    frames = list(frames)
    if not frames:
        return None
    prompts = list(prompts)
    if len(prompts) < len(frames):
        prompts += [prompts[-1] if prompts else ""] * (len(frames) - len(prompts))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "video"
    vid_dir = pathlib.Path(runs_dir) / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    path = vid_dir / (safe + ".mp4")
    h, w = np.asarray(frames[0]).shape[:2]
    cap_h, font = 40, cv2.FONT_HERSHEY_SIMPLEX
    fps = max(1, int(round(15 * float(speed))))  # MolmoBot runs ~15 Hz
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=None)
    try:
        for img, prompt in zip(frames, prompts):
            canvas = np.zeros((h + cap_h, w, 3), dtype=np.uint8)
            canvas[:h] = np.asarray(img)[:, :, :3]
            label = prompt or "—"
            if len(label) > 110:
                label = label[:107] + "..."
            cv2.putText(canvas, label, (8, h + 27), font, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
            writer.append_data(canvas)
    finally:
        writer.close()
    return path


def main():
    p = argparse.ArgumentParser(description="Interactive MolmoSpaces runner for MolmoBot")
    p.add_argument("--web-port", type=int, default=8888, help="web UI port")
    p.add_argument("--house-index", type=int, default=0, help="ProcTHOR-10k house index")
    p.add_argument("--split", default="val", help="house split (val/train/test)")
    p.add_argument("--instruction", default="", help="initial instruction")
    p.add_argument("--max-steps", type=int, default=600, help="policy steps before auto-pause")
    p.add_argument("--num-scenes", type=int, default=16, help="how many house indices to offer")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--no-bowl", dest="add_bowl", action="store_false",
                   help="don't attach the demo bowl receptacle on house 0")
    # Policy
    p.add_argument("--stub", action="store_true",
                   help="only offer the no-model wobble policy (local plumbing test, no GPU)")
    p.add_argument("--checkpoint", default="allenai/MolmoBot-DROID",
                   help="HF repo id or local dir for the real policy")
    p.add_argument("--action-type", default="joint_pos",
                   help="MolmoBot-DROID uses absolute joint_pos")
    args = p.parse_args()

    # The VLA selector only advertises what this server can actually run: MolmoBot
    # needs the `olmo` package (and a GPU); `stub` always works. --stub hides MolmoBot.
    available = []
    if molmobot_available() and not args.stub:
        available.append("MolmoBot")
    available.append("stub")
    logger.info("Available VLAs: %s", available)

    worker = RolloutWorker(args, available)
    worker.start()
    app = build_app(worker)
    logger.info("Web UI on :%d", args.web_port)
    app.run(host="0.0.0.0", port=args.web_port, threaded=True)


if __name__ == "__main__":
    main()
