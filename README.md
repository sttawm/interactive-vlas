# interactive-vlas

Interactive playgrounds for **vision-language-action policies (VLAs)**: type plain-English
instructions and watch a policy carry them out in a simulator, live in your browser. Change
the instruction mid-rollout to correct the robot or feed it staged subgoals.

The goal is to compare how steerable different VLAs are by language — one long prompt vs.
staged subgoals from a planner — across different policies and environments.

## Layout — one folder per VLA + environment

Each VLA+env pairing is a self-contained instance with its own `setup.sh`, `run.sh`, and app:

| Instance | Policy | Environment | Status |
|----------|--------|-------------|--------|
| [`pi05_libero/`](pi05_libero/) | π0.5 (OpenPI) | LIBERO | ✅ working |
| [`molmobot_molmospaces/`](molmobot_molmospaces/) | MolmoBot | MolmoSpaces (MuJoCo) | 🚧 rig validated (CPU); GPU run pending |
| [`molmoact2_maniskill/`](molmoact2_maniskill/) | MolmoAct2 | ManiSkill | 🗓️ planned |
| [`molmoact2_molmospaces/`](molmoact2_molmospaces/) | MolmoAct2 | MolmoSpaces (MuJoCo) | 🗓️ planned |

Code shared across the Molmo instances lives in [`molmo_shared/`](molmo_shared/). The
upstream reference repos (MolmoSpaces, MolmoBot, MolmoAct2) are cloned under `molmo/`,
which is gitignored — they are not vendored.

To get started, `cd` into an instance folder and follow its README, e.g.:

```bash
cd pi05_libero
./setup.sh    # one-time, on a fresh GPU pod
./run.sh      # starts the policy server + interactive web UI
```

See [`pi05_libero/README.md`](pi05_libero/README.md) for the full quickstart, GPU
requirements, usage, and interaction modes.

## Conventions shared across instances

- **Policy-server / sim-client split** — the VLA runs as a server on the GPU; the simulator
  is a client that streams observations in and actions out. The interactive web UI wraps the
  client loop and exposes the policy's language prompt as a live, user-controlled variable.
- **Browser viewer** — the rendered camera view is polled as frames (robust through reverse
  proxies like RunPod's), with a text box to type/replace instructions.
- **`runs/` logging** — every rollout records the typed instructions, a video, the actions,
  and metadata (scene, seed, checkpoint/commit, success).
