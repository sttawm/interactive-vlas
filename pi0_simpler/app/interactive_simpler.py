"""π0 (Bridge, rephrase-finetuned) + SimplerEnv backend for the shared interactive web UI.

This is the SimplerEnv/π0-specific half: it owns the SimplerEnv (SAPIEN/ManiSkill2) WidowX
simulator, the LeRobot ``PI0Policy`` (run **in-process**, PyTorch/GPU — there is no separate
policy server here, unlike pi05_libero), and the rollout thread. All UI / routing / video
composition is generic and lives in ``shared/webui.py``.

Reuses the CoVer/INT-ACT eval code (``run_simpler_eval_with_openpi.py`` and friends) for the
policy load, observation adapter, action post-processing, task suite, and the contrastive
verifier — the only real change is that the prompt is a **live, user-controlled variable** you
can edit mid-rollout. See CoVer: "Scaling Verification Can Be More Effective than Scaling Policy
Learning for Vision-Language-Action Alignment" (arXiv 2602.12281).

Two loops:
  - plain (default): single live prompt, deterministic π0 (``noise_std=0``), immediate replan on edit.
  - verifier (``--verifier``): the CoVer loop — sample ``policy_batch_inference_size`` actions per
    prompt across ``lang_rephrase_num`` rephrases of the current prompt, score with the verifier,
    auto-select the best prompt+action chunk. Rephrases come from the shipped ``ert_rephrases``
    JSON for canonical task prompts, else are generated live via Claude (``app/rephrase.py``).

Runs in the CoVer ``.venv_cover`` (Python 3.10). Set the CoVer paths on sys.path via the
``COVER_DIR`` / ``COVER_INFERENCE`` env vars (run.sh does this) so the ``experiments.*`` and
``bridge_verifier`` imports resolve.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import os
import pathlib
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import cv2
import numpy as np
import imageio

# Make the repo-root `shared` package + this app dir importable when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from shared import webui  # noqa: E402
import rephrase  # noqa: E402  (sibling module, app/ dir on sys.path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pi0_simpler")

# Default policy: π0 finetuned on Bridge V2 *with* paraphrase augmentation (INT-ACT release,
# reused by CoVer). Swap to "juexzz/INTACT-pi0-finetune-bridge" for the no-paraphrase baseline.
DEFAULT_CHECKPOINT = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

# SimplerEnv WidowX/Bridge task suites (from CoVer's simpler_benchmark.task_map). suite -> tasks.
SUITES = collections.OrderedDict([
    ("simpler_widowx", [
        "widowx_put_eggplant_in_basket",
        "widowx_spoon_on_towel",
        "widowx_stack_cube",
        "widowx_carrot_on_plate",
    ]),
    ("simpler_ood", [
        "widowx_redbull_on_plate",
        "widowx_zucchini_on_towel",
        "widowx_tennis_ball_in_basket",
    ]),
])

# task_name -> (pretty label, canonical instruction). The canonical string is the SimplerEnv
# task's own language instruction (== env.get_language_instruction()); it doubles as the key into
# the shipped rephrase JSON. Used for the cheap /config display; the true canonical is re-read
# from the env at reset. Kept in sync with CoVer's tasks.
TASK_META = {
    "widowx_put_eggplant_in_basket": ("Eggplant in basket", "put eggplant into yellow basket"),
    "widowx_spoon_on_towel":         ("Spoon on towel",     "put the spoon on the towel"),
    "widowx_stack_cube":             ("Stack cube",         "stack the green block on the yellow block"),
    "widowx_carrot_on_plate":        ("Carrot on plate",    "put carrot on plate"),
    "widowx_redbull_on_plate":       ("Redbull on plate (OOD)",   "put redbull can on plate"),
    "widowx_zucchini_on_towel":      ("Zucchini on towel (OOD)",  "put the zucchini on the towel"),
    "widowx_tennis_ball_in_basket":  ("Tennis ball in basket (OOD)", "put tennis ball into yellow basket"),
}


def _rephrase_json_path():
    """Locate the shipped ert_rephrases JSON in the CoVer clone (COVER_INFERENCE env or default)."""
    inf = os.environ.get("COVER_INFERENCE")
    if inf:
        p = pathlib.Path(inf) / "experiments/robot/simpler/simpler_rephrased_final_eval_vlm.json"
        if p.exists():
            return p
    cover = os.environ.get("COVER_DIR", "/workspace/cover-vla")
    return pathlib.Path(cover) / "CoVer_VLA/inference/experiments/robot/simpler/simpler_rephrased_final_eval_vlm.json"


class SimplerWorker(threading.Thread):
    """Implements the shared.webui worker contract for π0-Bridge + SimplerEnv."""

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args

        # Rephrases keyed by canonical instruction (shipped JSON); loaded eagerly (cheap, no torch).
        self._rephrase_json = {}
        try:
            with open(_rephrase_json_path()) as fh:
                self._rephrase_json = json.load(fh).get("instructions", {})
        except Exception:
            logger.warning("rephrase JSON not found (chips/verifier-rephrases limited); path=%s",
                           _rephrase_json_path())
        self._rephrase_cache = {}  # prompt -> [rephrase, ...] (JSON hit or live Claude), memoised

        # Lazily-loaded heavy backends (torch / simpler_env / policy / verifier).
        self._policy = None
        self._adapter = None
        self._image_key = None
        self._verifier = None
        self._get_image = None
        self._simpler_make = None
        self._convert_action = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._instruction = ""
        self._latest_jpeg = _placeholder_jpeg("starting…")
        self._paused = True
        self._reset_to = None
        self._clear_plan = False

        self._st = {  # raw status fields; snapshot_status() formats these for display
            "suite": next(iter(SUITES)), "task": "", "task_language": "", "instruction": "",
            "sent_prompt": "", "step": 0, "step_limit": args.max_rollout_steps,
            "paused": True, "limit_reached": False, "success": False, "connected": False,
            "verifier": bool(args.verifier), "vscore": None, "selected": "", "phrase_scores": [],
        }
        self._record_frames = []
        self._record_prompts = []
        self._record_actions = []
        self._run_dir = None
        self._run_meta = {}
        self._action_history = []  # verifier past-action buffer (reset per episode / on replan)

    # ----- contract: config / status / control -----

    def config(self):
        options_by = {}
        for suite, tasks in SUITES.items():
            opts = []
            for tname in tasks:
                label, canonical = TASK_META.get(tname, (tname, ""))
                entry = self._rephrase_json.get(canonical, {})
                examples = entry.get("ert_rephrases", [])[: max(0, self.args.lang_rephrase_num)]
                opts.append({"value": tname, "label": label,
                             "default_prompt": canonical, "examples": examples})
            options_by[suite] = opts
        title = "π0 (rephrase) · SimplerEnv"
        if self.args.verifier:
            title += " + CoVer"
        return {
            "title": title,
            "selectors": [
                {"name": "suite", "label": "Suite", "options": list(SUITES.keys())},
                {"name": "task", "label": "Task", "depends_on": "suite", "options_by": options_by},
            ],
            "instruction_label": "Instruction to π0 — blank uses the task's default",
            "instruction_placeholder": "e.g. put the spoon on the towel",
        }

    def default_prompt(self, selection):
        tname = selection.get("task") or ""
        return TASK_META.get(tname, (tname, ""))[1]

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
            state = "⏹ episode ended — Reset"
        elif s["paused"]:
            state = "⏸ paused"
        else:
            state = "▶ running"
        out = {
            "Step": "%d / %d" % (s["step"], s["step_limit"]),
            "State": state,
            "Verifier": "on" if s["verifier"] else "off",
            "paused": s["paused"],
            "limit_reached": s["limit_reached"],
        }
        if s["verifier"] and s["vscore"] is not None:
            out["Score"] = "%.3f" % s["vscore"]
            if s["selected"]:
                out["Selected"] = s["selected"] if len(s["selected"]) < 48 else s["selected"][:45] + "…"
            # Per-phrase verifier scores (best first); the ➤ marks the executed phrase. Only
            # interesting when >1 phrase was scored (verifier ran over rephrases).
            ps = s.get("phrase_scores") or []
            if len(ps) > 1:
                for rank, (phrase, score) in enumerate(ps, 1):
                    mark = "➤" if phrase == s["selected"] else "  "
                    label = phrase if len(phrase) < 44 else phrase[:41] + "…"
                    out["%s %d. %s" % (mark, rank, label)] = "%.3f" % score
        return out

    def save_video(self, name, speed=1.0):
        with self._lock:
            frames = list(self._record_frames)
            prompts = list(self._record_prompts)
        return webui.compose_video(frames, prompts, name, speed=speed, runs_dir=self.args.runs_dir)

    def stop(self):
        self._stop.set()

    # ----- rephrase sourcing (verifier path) -----

    def _rephrases_for(self, prompt):
        """The rephrase pool the verifier scores over, minus the base prompt. Shipped JSON when
        `prompt` is a canonical task instruction, else generated live via Claude; both cached."""
        k = max(0, self.args.lang_rephrase_num - 1)
        if k == 0:
            return []
        if prompt in self._rephrase_cache:
            return self._rephrase_cache[prompt]
        entry = self._rephrase_json.get(prompt)
        if entry and entry.get("ert_rephrases"):
            out = list(entry["ert_rephrases"])[:k]
        else:
            logger.info("No shipped rephrases for %r — generating %d via Claude…", prompt, k)
            out = rephrase.generate(prompt, k)  # [] if no ANTHROPIC_API_KEY / on error
        self._rephrase_cache[prompt] = out
        return out

    # ----- recording -----

    def _start_new_run(self, suite, task):
        self._finalize_run()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = pathlib.Path(self.args.runs_dir) / ("%s_%s" % (stamp, task))
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._record_frames = []
        self._record_prompts = []
        self._record_actions = []
        self._run_meta = {"suite": suite, "task": task, "started": stamp,
                          "checkpoint": self.args.checkpoint, "verifier": bool(self.args.verifier),
                          "cover_commit": os.environ.get("COVER_COMMIT", "")}

    def _finalize_run(self):
        if not self._run_dir:
            return
        try:
            if self._record_frames:
                imageio.mimwrite(str(self._run_dir / "rollout.mp4"),
                                 [np.asarray(f) for f in self._record_frames], fps=10)
            if self._record_actions:
                np.save(str(self._run_dir / "actions.npy"), np.asarray(self._record_actions))
            with open(self._run_dir / "instructions.txt", "w") as fh:
                last = None
                for i, p in enumerate(self._record_prompts):
                    if p != last:
                        fh.write("step=%d\t%s\n" % (i, p))
                        last = p
            with open(self._run_dir / "metadata.json", "w") as fh:
                meta = dict(self._run_meta)
                meta["final_step"] = self._st["step"]
                meta["success"] = self._st["success"]
                json.dump(meta, fh, indent=2)
        except Exception:
            logger.exception("finalize_run failed")
        finally:
            self._run_dir = None

    # ----- backends (lazy; heavy imports live here so config()/--stub need no GPU) -----

    def _load_backends(self):
        if self._policy is not None or self.args.stub:
            return
        import torch  # noqa: F401
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        from experiments.robot.simpler.eval_utils import (
            create_bridge_adapter_wrapper, convert_maniskill_with_bridge_adapter)

        logger.info("Loading π0 policy from %s …", self.args.checkpoint)
        policy = PI0Policy.from_pretrained(self.args.checkpoint)
        if torch.cuda.is_available():
            policy.to("cuda")
            policy.config.device = "cuda"
        policy.config.n_action_steps = int(self.args.n_action_steps)
        self._policy = policy
        self._image_key = list(policy.config.image_features.keys())[0]
        self._adapter = create_bridge_adapter_wrapper(self.args.action_ensemble_temp)
        self._get_image = get_image_from_maniskill2_obs_dict
        self._simpler_make = simpler_env.make
        self._convert_action = convert_maniskill_with_bridge_adapter
        self._torch = torch

        if self.args.verifier:
            from bridge_verifier.ensemble_eval import EfficientEnsembleMerged
            from experiments.robot.simpler.eval_utils import process_inputs, process_raw_image_to_jpg
            cover = os.environ.get("COVER_DIR", "/workspace/cover-vla")
            ckpt = os.environ.get("VERIFIER_CKPT",
                                  str(pathlib.Path(cover) / "bridge_verifier" / "cover_verifier_bridge.pt"))
            logger.info("Loading CoVer verifier from %s …", ckpt)
            self._verifier = EfficientEnsembleMerged(ckpt)
            self._process_inputs = process_inputs
            self._process_raw_image_to_jpg = process_raw_image_to_jpg
        logger.info("Backends ready (image_key=%s, verifier=%s).", self._image_key, self.args.verifier)

    # ----- the rollout loop -----

    def run(self):
        try:
            self._load_backends()
        except Exception:
            logger.exception("backend load failed")
            with self._lock:
                self._latest_jpeg = _placeholder_jpeg("backend load failed — see logs")
            return

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
                    env = None
                suite = reset_to.get("suite", next(iter(SUITES)))
                task = reset_to.get("task") or SUITES.get(suite, [""])[0]
                logger.info("Loading SimplerEnv task %s …", task)
                try:
                    env, task_language, obs = self._make_env(task)
                except Exception:
                    logger.exception("env load failed")
                    with self._lock:
                        self._latest_jpeg = _placeholder_jpeg("env load failed — see logs")
                    env = None
                    continue
                action_plan.clear()
                self._reset_verifier_state()
                step = 0
                self._start_new_run(suite, task)
                with self._lock:
                    self._st.update(suite=suite, task=task, task_language=str(task_language),
                                    instruction=self._instruction, step=0, sent_prompt="",
                                    success=False, limit_reached=False, connected=True,
                                    vscore=None, selected="")
                self._publish_frame(obs, env, self._instruction)
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
                self._reset_verifier_state()
            try:
                obs, done, trunc = self._step_once(env, obs, action_plan, instruction, step)
                step += 1
            except Exception:
                logger.exception("step failed; pausing")
                self.set_paused(True)
                continue

            reached_limit = done or trunc or (step >= self.args.max_rollout_steps)
            with self._lock:
                self._st["step"] = step
                self._st["success"] = bool(done)
                if reached_limit:
                    self._st["limit_reached"] = True
                    self._st["paused"] = True
                    self._paused = True
            if reached_limit:
                logger.info("Episode ended (success=%s, trunc=%s, step=%d)", done, trunc, step)
                self._finalize_run()
        self._finalize_run()

    def _make_env(self, task):
        if self.args.stub:
            return _StubEnv(task), TASK_META.get(task, (task, task))[1], None
        env = self._simpler_make(task)
        obs, _ = env.reset(seed=self.args.seed)
        # Optional settle steps (SimplerEnv Bridge usually needs none).
        for _ in range(self.args.num_steps_wait):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
        try:
            lang = env.get_language_instruction()
        except Exception:
            lang = TASK_META.get(task, (task, task))[1]
        return env, lang, obs

    def _reset_verifier_state(self):
        self._action_history = []

    def _step_once(self, env, obs, action_plan, instruction, step):
        if self.args.stub:
            return self._step_stub(env, obs, instruction), False, False

        raw_img = self._get_image(env, obs)  # RGB uint8 (H,W,3)
        processed = self._adapter.preprocess({
            "observation.images.top": raw_img,
            "observation.state": obs,
            "task": str(instruction),
        })
        dev = self._torch.device(self._policy.config.device)
        img_t = processed["observation.images.top"]
        state_t = processed["observation.state"]
        img_t = img_t.to(dev) if hasattr(img_t, "to") else img_t
        state_t = state_t.to(dev) if hasattr(state_t, "to") else state_t

        if self.args.verifier:
            execute_action = self._infer_verifier(action_plan, img_t, state_t, raw_img, instruction)
        else:
            execute_action = self._infer_plain(action_plan, img_t, state_t, instruction)

        with self._lock:
            self._st["sent_prompt"] = str(instruction)
        obs, reward, done, trunc, info = env.step(execute_action)

        self._record_frames.append(raw_img)
        self._record_prompts.append(str(self._st.get("selected") or instruction))
        self._record_actions.append(np.asarray(execute_action))
        self._publish_frame(obs, env, self._st.get("selected") or instruction)
        return obs, bool(done), bool(trunc)

    def _infer_plain(self, action_plan, img_t, state_t, instruction):
        """Single deterministic prompt; refill an n_action_steps chunk when the plan drains."""
        if not action_plan:
            observation = {
                self._image_key: img_t,
                "observation.state": state_t,
                "task": [str(instruction)],
            }
            with self._torch.no_grad():
                q = self._policy.select_action(observation, noise_std=0.0)
                action_plan.extend(list(q))
                q.clear()
        single = action_plan.popleft().detach().cpu().numpy()  # (batch=1, 7)
        return self._convert_action(single[0:1], verifier_action=False,
                                    action_ensemble_temp=self.args.action_ensemble_temp)

    def _infer_verifier(self, action_plan, img_t, state_t, raw_img, instruction):
        """CoVer loop: sample B actions per prompt over the prompt + its rephrases, score EACH
        phrase with the verifier, pick the best. Ports the mechanics of run_simpler_eval_with_openpi.py's
        verifier branch (process_inputs, gripper voting, action-chunk carry) but scores per phrase —
        rather than the paper's original-first shortcut — so every phrase's score can be shown live."""
        B = int(self.args.policy_batch_inference_size)
        base = str(instruction)
        rephrases = self._rephrases_for(base)
        unique_prompts = [base] + list(rephrases)
        batch_size = B * len(unique_prompts)
        task_list = []
        for p in unique_prompts:
            task_list.extend([p] * B)

        if not action_plan:
            observation = {
                self._image_key: img_t.repeat(batch_size, 1, 1, 1),
                "observation.state": state_t.repeat(batch_size, 1),
                "task": task_list,
            }
            with self._torch.no_grad():
                q = self._policy.select_action(observation, noise_std=1.0)
                predefined = list(q)   # n_action_steps tensors, each (batch_size, 7)
                q.clear()

            num_past = min(len(self._action_history), 6)
            hist = self._process_inputs(batch_size, predefined, verifier_action=True,
                                        action_history=list(self._action_history), cfg=self.args)
            exec_hist = self._process_inputs(batch_size, predefined, verifier_action=False,
                                             action_history=list(self._action_history), cfg=self.args)
            img_jpg = self._process_raw_image_to_jpg(raw_img)

            # Score each phrase's B action samples separately → one score per phrase.
            phrase_scores = []                 # [(phrase, best_score)]
            best = None                        # (score, global_idx, action_history_for_verifier, phrase)
            for i, prompt in enumerate(unique_prompts):
                s0 = i * B
                with self._torch.no_grad():
                    score, _sel, ahist, local_idx = self._verifier.compute_max_similarity_scores_batch(
                        images=[img_jpg] * B, instructions=[prompt] * B,
                        all_action_histories=hist[s0:s0 + B], cfg_repeat_language_instructions=B)
                score = float(score)
                gidx = s0 + int(local_idx)
                phrase_scores.append((prompt, score))
                if best is None or score > best[0]:
                    best = (score, gidx, ahist, prompt)

            best_score, gidx, max_hist, selected = best
            execute_action = exec_hist[gidx][num_past].copy()
            # gripper voting within the selected phrase's B samples
            gstart = (gidx // B) * B
            grippers = np.stack(exec_hist[gstart:gstart + B])[:, num_past, -1]
            close_votes, open_votes = int((grippers >= 0).sum()), int((grippers < 0).sum())
            if close_votes > open_votes:
                execute_action[-1] = 1.0
            elif open_votes > close_votes:
                execute_action[-1] = -1.0
            execute_action[-1] = float(np.sign(execute_action[-1]))
            self._action_history.append(max_hist[num_past].copy())

            # queue the remaining timesteps of the selected batch item
            for ts in range(1, int(self.args.n_action_steps)):
                action_plan.append(predefined[ts][gidx:gidx + 1])
            with self._lock:
                self._st["vscore"] = best_score
                self._st["selected"] = selected
                self._st["phrase_scores"] = sorted(phrase_scores, key=lambda x: -x[1])
            return execute_action

        single = action_plan.popleft().detach().cpu().numpy()  # (1,7)
        self._action_history.append(self._convert_action(
            single[0:1], verifier_action=True, action_ensemble_temp=self.args.action_ensemble_temp))
        return self._convert_action(single[0:1], verifier_action=False,
                                    action_ensemble_temp=self.args.action_ensemble_temp)

    # ----- stub (no GPU / no sim: validates the web contract only) -----

    def _step_stub(self, env, obs, instruction):
        env.t += 1
        time.sleep(0.05)  # no real sim latency in stub — pace it so the placeholder animates
        return None

    def _publish_frame(self, obs, env, overlay=None):
        DS = 384
        if self.args.stub or obs is None:
            frame = _wave_frame(getattr(env, "t", 0), DS)
        else:
            frame = np.ascontiguousarray(self._get_image(env, obs))
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


class _StubEnv:
    """Minimal env stand-in for --stub: no sim, no policy — just an animated placeholder so the
    shared web UI / selectors / play-pause can be exercised on a CPU-only box."""
    def __init__(self, task):
        self.task = task
        self.t = 0

    def close(self):
        pass


def _wave_frame(t, ds):
    x = np.linspace(0, 4 * np.pi, ds)
    y = np.linspace(0, 4 * np.pi, ds)
    xx, yy = np.meshgrid(x, y)
    v = (np.sin(xx + t * 0.15) + np.cos(yy + t * 0.1))
    img = np.zeros((ds, ds, 3), dtype=np.uint8)
    img[..., 0] = ((v + 2) / 4 * 120 + 30).astype(np.uint8)
    img[..., 1] = 40
    img[..., 2] = ((2 - v) / 4 * 120 + 30).astype(np.uint8)
    cv2.putText(img, "STUB (no policy/sim)", (16, ds - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1)
    return img


def _placeholder_jpeg(text):
    img = np.zeros((384, 384, 3), dtype=np.uint8)
    cv2.putText(img, text, (16, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def main():
    p = argparse.ArgumentParser(description="Interactive π0-Bridge + SimplerEnv (shared web UI)")
    p.add_argument("--web-port", type=int, default=8888, help="web UI port")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF id for LeRobot PI0Policy")
    p.add_argument("--verifier", action="store_true", help="run the CoVer verifier loop")
    p.add_argument("--lang-rephrase-num", type=int, default=8, help="K: prompt + K-1 rephrases (verifier)")
    p.add_argument("--policy-batch-inference-size", type=int, default=5, help="action samples per prompt (verifier)")
    p.add_argument("--n-action-steps", type=int, default=4, help="action chunk length / replan period")
    p.add_argument("--action-ensemble-temp", type=float, default=-0.8)
    p.add_argument("--max-rollout-steps", type=int, default=600, help="hard step cap before auto-pause")
    p.add_argument("--num-steps-wait", type=int, default=0, help="no-op settle steps after reset")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--stub", action="store_true", help="no policy/sim — animated placeholder (CPU plumbing test)")
    args = p.parse_args()

    worker = SimplerWorker(args)
    worker.start()
    app = webui.build_app(worker)
    logger.info("Web UI on :%d (checkpoint=%s, verifier=%s, stub=%s)",
                args.web_port, args.checkpoint, args.verifier, args.stub)
    app.run(host="0.0.0.0", port=args.web_port, threaded=True)


if __name__ == "__main__":
    main()
