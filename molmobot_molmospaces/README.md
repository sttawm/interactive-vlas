# molmobot_molmospaces

Type plain-English instructions and watch **MolmoBot** carry them out in the
**MolmoSpaces** MuJoCo simulator — live, in your browser. Change the instruction
mid-rollout to correct the robot or feed it staged subgoals.

```
   browser ──▶ load a ProcTHOR house, type "put the salt shaker in the bowl", watch it act
                │
   ┌────────────┴───────────── GPU pod ──────────────────────────┐
   │  web UI (:8888)                                              │
   │  MolmoSpaces MuJoCo sim  +  MolmoBot VLA  (both in-process)  │
   │  live prompt = obs["task"]      allenai/MolmoBot-DROID       │
   └─────────────────────────────────────────────────────────────┘
```

Unlike [`pi05_libero/`](../pi05_libero/) there is **no separate policy server** — MolmoBot
is a ~4B in-process model (this is MolmoBot's own `demo_policy.ipynb` loop, made interactive),
so the sim and the model share one process and one GPU.

---

## Setup (one command on a fresh GPU pod)

**You need:** an NVIDIA GPU pod with **≥16 GB VRAM**, Ubuntu 22.04, Python 3.11, and
~60 GB disk (checkpoint ~8–9 GB + scene assets). A [RunPod](https://runpod.io) RTX
A5000/4090/A6000 works. Expose **port 8888** (HTTP) and **22** (SSH) when creating the pod.

```bash
git clone https://github.com/sttawm/interactive-pi.git
cd interactive-pi/molmobot_molmospaces
./setup.sh        # clones MolmoBot, installs it + pinned molmo_spaces, downloads MolmoBot-DROID
./run.sh          # starts the interactive web UI on :8888
```

`setup.sh` installs everything under `/workspace` (clone in `/workspace/MolmoBot`, caches in
`/workspace/.cache`) so it survives pod restarts. Re-running is safe.

### Open the UI from your laptop

- **RunPod HTTP proxy (easiest):** `https://<POD_ID>-8888.proxy.runpod.net`
- **SSH tunnel:** `ssh -L 8888:localhost:8888 root@<ip> -p <port>` then `http://localhost:8888`

---

## Using it

1. **Load a scene** — pick a ProcTHOR-10k house index (start with **0**, the demo kitchen
   that ships with a bowl) and hit **Load scene & start**.
2. **Type an instruction** and hit **Send** (or Enter). MolmoBot starts acting toward it.
3. **Watch** the live view — left is the exo (3rd-person) camera, right is the wrist camera
   (exactly the two images the policy sees).
4. **Steer it** — send a new instruction any time. It forces an immediate replan (the action
   buffer is flushed), so use it for corrections or staged subgoals.

> ⚠️ Language alone isn't magic: an instruction only works if the **objects actually exist in
> the loaded house**. Compose instructions from things you can see in the camera view.

### Knobs (env vars for `run.sh`)

| Var | Default | Meaning |
|-----|---------|---------|
| `HOUSE` | `0` | ProcTHOR-10k house index to start on |
| `SPLIT` | `val` | house split (`val`/`train`/`test`) |
| `WEB_PORT` | `8888` | web UI port |
| `CHECKPOINT` | `allenai/MolmoBot-DROID` | HF repo id or local checkpoint dir |
| `STUB` | `0` | `1` = no-model wobble policy (see below) |

---

## Testing the rig without a GPU (`STUB=1`)

The whole harness — scene build, dual-camera rendering, browser stream, instruction
plumbing, `runs/` logging — runs on a **CPU/Mac with no model**:

```bash
# in an env that has molmo_spaces[mujoco] + flask + opencv (e.g. the `mlspaces` conda env)
STUB=1 ./run.sh
# or directly:
python app/interactive_molmospaces.py --stub --web-port 8888
```

The stub policy is a scripted joint wobble — it **ignores the instruction** (a scripted
policy can't be steered by language), so it only proves the plumbing. Swap to the real
policy on a GPU pod to actually steer by typed language. This is how the app was validated
locally before any GPU run.

> Note on `molmo_spaces` versions: MolmoBot pins an older `molmo_spaces` (the one the demo
> targets); a fresh clone of `molmospaces` is newer and changed `add_robot_to_scene`'s
> signature. The scene builder detects and supports both. If other API drift surfaces on the
> pod, it'll be a small fix in `app/interactive_molmospaces.py::SimWorld`.

---

## What gets logged

Every loaded scene starts a new run under `runs/`:

```
runs/2026-06-25_125715_house0/
  instructions.txt   # every instruction typed, with timestamp + step
  rollout.mp4        # video (exo | wrist stacked, what the policy sees)
  actions.npy        # the executed action dicts (object array)
  metadata.json      # house, split, policy, checkpoint, action_type, seed
```

Pull them to your laptop with `scp`/`rsync` from the pod.

---

## How the live instruction is wired

The single hook is `obs["task"]`, set every policy step from the user-typed string
(`RolloutWorker.run` → `SimWorld.obs`). MolmoBot's `RealRobotVLAPolicy` reads `obs["task"]`
each time it refills its action buffer (every `execute_horizon` steps). To make edits land
immediately, sending a new instruction also flushes the buffer
(`RolloutWorker._do_force_refresh`). See [`molmo_shared/`](../molmo_shared/) for the pattern
shared with the other instances.
