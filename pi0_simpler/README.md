# pi0_simpler — interactive π0 (Bridge) + SimplerEnv

Type plain-English instructions and watch **π0 finetuned on Bridge** carry them out in the
**SimplerEnv** (SIMPLER) WidowX simulator — live, in your browser. Change the instruction
mid-rollout to steer the robot. Optionally run the **CoVer verifier** in the loop.

The policy is π0 finetuned on BridgeData V2 with paraphrase augmentation
(`juexzz/INTACT-pi0-finetune-rephrase-bridge`, from the INT-ACT release), loaded via **LeRobot's
`PI0Policy`** and run **in-process** with the sim (no separate policy server). The verifier is from
**CoVer** — *"Scaling Verification Can Be More Effective than Scaling Policy Learning for
Vision-Language-Action Alignment"* (Kwok, Zhang, …, Finn, Pavone; arXiv 2602.12281,
[cover-vla.github.io](https://cover-vla.github.io)). We reuse CoVer's SimplerEnv eval code
(`run_simpler_eval_with_openpi.py` and friends) and only make the prompt a live variable.

```
   browser ──▶ pick a WidowX task, type "put the spoon on the towel", watch it act
                │
   ┌────────────┴───────────────── GPU pod ────────────────────────────┐
   │  web UI (:8888)  ──in-process──▶  π0 (LeRobot PI0Policy, PyTorch)  │
   │  SimplerEnv (SAPIEN/ManiSkill2) sim + live prompt                  │
   │  optional: CoVer contrastive verifier (bridge_verifier)           │
   └───────────────────────────────────────────────────────────────────┘
```

This instance serves the **same web page** as the others via [`shared/webui.py`](../shared/).

---

## Setup (one command on a fresh GPU pod)

**You need:** an NVIDIA GPU pod with **working Vulkan** (SimplerEnv/SAPIEN renders with Vulkan,
not just EGL/OSMesa — see Troubleshooting), Ubuntu 22.04, **Python 3.10**, CUDA 12.x, ~60 GB disk.
**Expose port 8888 (HTTP) and port 22 (SSH)** when creating the pod.

```bash
git clone https://github.com/sttawm/interactive-vlas.git
cd interactive-vlas/pi0_simpler
./setup.sh        # clones CoVer, builds .venv_cover, prefetches π0 + verifier ckpts (~15-25 min)
./run.sh          # starts the web UI in tmux; prints the URL
```

Everything heavy (the CoVer clone, `.venv_cover`, the checkpoints) installs under the persistent
**`/workspace`** volume. `run.sh` launches in **tmux**, so you can disconnect SSH.

### Open the UI from your laptop

- **RunPod proxy:** `https://<POD_ID>-8888.proxy.runpod.net` (`echo $RUNPOD_POD_ID`).
- **SSH tunnel:** `ssh -L 8888:localhost:8888 root@<ip> -p <port>` then `http://localhost:8888`.

### Restart (after a pod stop/start)

`./restart.sh` — re-runs `setup.sh` (idempotent; repairs container-disk bits, skips the big
downloads/venv build) then `run.sh`, in a detached tmux `boot` session.

### Stop (to save GPU credits)

`tmux kill-session -t webapp` then stop the pod. The install persists on `/workspace`.

---

## Using it

1. **Pick a task** — a suite (`simpler_widowx` = the 4 in-distribution WidowX tasks; `simpler_ood`
   = the OOD tasks) and a task. The task's canonical instruction prefills the box; the red-teamed
   `ert_rephrases` from the paper show up as clickable example chips.
2. **Press Play.** The WidowX arm acts toward the instruction. You watch the live overhead camera.
3. **Steer it** — edit the instruction any time (paraphrase, correction, or a different in-scene
   goal). It triggers an **immediate replan**.

> ⚠️ Language only works if the objects/goal exist in the chosen scene — SimplerEnv WidowX tasks
> are fixed tabletop setups (eggplant/basket, spoon/towel, blocks, carrot/plate, …).

### Tasks

| Suite | Tasks |
|-------|-------|
| `simpler_widowx` (ID) | eggplant in basket · spoon on towel · stack cube · carrot on plate |
| `simpler_ood` | redbull on plate · zucchini on towel · tennis ball in basket |

### Two modes

- **Plain (default):** deterministic π0 on your single live prompt. Best for testing how a
  reworded instruction changes behavior.
- **Verifier (`VERIFIER=1 ./run.sh`):** the **CoVer** loop — π0 samples several actions across
  your prompt **and rephrases of it**, the verifier scores them, and the best prompt+action is
  auto-selected. The Status panel shows the verifier **Score** and the **Selected** instruction
  (which can differ from what you typed — that's the verifier steering).
  - Rephrases come from the paper's shipped `ert_rephrases` JSON when your prompt matches one of
    the canonical task instructions; for a **custom** prompt they're generated live via Claude
    (needs `ANTHROPIC_API_KEY`; falls back to just your prompt if unset). Set `REPHRASE_MODEL`
    to override the model. `LANG_REPHRASE_NUM` / `BATCH` tune the ensemble size.

---

## What gets logged

Every loaded task starts a run under `runs/`:

```
runs/2026-07-06_140312_widowx_spoon_on_towel/
  instructions.txt   # every instruction (or verifier-selected instruction), with step
  rollout.mp4        # the overhead camera view
  actions.npy        # executed actions (N, 7)
  metadata.json      # suite, task, checkpoint, verifier on/off, CoVer commit, success
```

Pull them locally with `rsync` (see the top-level README).

---

## How it works (and what was changed from stock CoVer)

CoVer's `run_simpler_eval_with_openpi.py` already loads `PI0Policy`, builds SimplerEnv tasks, and
(optionally) runs the verifier. We reuse its modules verbatim — the policy load, the
`BridgeSimplerAdapter` preprocessing, `convert_maniskill_with_bridge_adapter`, `process_inputs`,
the task suite, and `EfficientEnsembleMerged` — and change one thing: the **prompt is a live,
user-controlled variable**, so you can edit it mid-episode (an edit clears the action chunk and
forces a replan). The plain loop uses deterministic sampling (`noise_std=0`); the verifier loop
uses the paper's sampling (`noise_std=1`, batch × rephrases).

- `app/interactive_simpler.py` — the worker (owns the sim + π0) + implements the `shared/webui.py`
  contract. `--stub` runs an animated placeholder with no policy/sim (CPU plumbing test).
- `app/rephrase.py` — live Claude rephraser for custom prompts (verifier mode).
- `setup.sh` / `run.sh` / `restart.sh` — reproducible install + tmux launch.

---

## Troubleshooting

- **SAPIEN / Vulkan render failures** (`No Vulkan extensions found…`, blank/black frames): this is
  the SimplerEnv-on-a-pod snag. SimplerEnv's WidowX rendering needs a working **Vulkan ICD**. Make
  sure `libvulkan1` is installed (setup does this) and the NVIDIA ICD is present — `vulkaninfo`
  should list your GPU. If your pod exposes an NVIDIA ICD at a nonstandard path, set
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`. The `setup.sh` render smoke test
  tells you immediately whether this pod can render.
- **OOM:** π0 + SAPIEN share the GPU; the verifier adds a second model + batched sampling. Lower
  `BATCH` / `LANG_REPHRASE_NUM`, or run plain (no `VERIFIER=1`).
- **Blank/frozen video over the RunPod proxy:** the UI polls frames; a frozen frame usually means
  paused or an episode that hit its horizon — check the Status panel, or use the SSH tunnel.

## Attribution

- Policy: [`juexzz/INTACT-pi0-finetune-rephrase-bridge`](https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge)
  (INT-ACT, *From Intention to Execution*, arXiv:2506.09930).
- Verifier + SimplerEnv eval code: [cover-vla/cover-vla](https://github.com/cover-vla/cover-vla)
  (CoVer, arXiv:2602.12281). SimplerEnv: [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv).
