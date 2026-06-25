# interactive-pi

Type plain-English instructions and watch **π0.5** (pi0.5, via [OpenPI](https://github.com/Physical-Intelligence/openpi)) carry them out in the **LIBERO** simulator — live, in your browser. Change the instruction mid-rollout to correct the robot or feed it staged subgoals.

```
   browser ──▶ pick a scene, type "put the bowl on the plate", watch it act
                │
   ┌────────────┴───────── GPU pod ─────────────────────────┐
   │  web UI (:8888)  ──websocket──▶  pi0.5 policy server (:8000)  │
   │  LIBERO sim + live prompt        gs://openpi-assets/pi05_libero │
   └────────────────────────────────────────────────────────┘
```

---

## Setup (one command on a fresh GPU pod)

**You need:** an NVIDIA GPU pod with **≥12 GB VRAM** (tested on an RTX A4500, 20 GB), Ubuntu 22.04, Python 3.11, and ~60 GB disk. A [RunPod](https://runpod.io) Community-Cloud RTX 3090/A4500/A5000 (~$0.20–0.30/hr) is plenty. **Expose port 8888 (HTTP) and port 22 (SSH)** when creating the pod. Prefer a pod **near you** — an EU pod viewed from the US is laggy (both the ~5-8 min startup and the live frame rate).

```bash
git clone https://github.com/sttawm/interactive-vlas.git
cd interactive-vlas/pi05_libero
./setup.sh        # one-time: OpenPI + LIBERO + GL libs + tmux + pi0.5 checkpoint (~15–20 min)
./run.sh          # starts the policy server + web UI, both in tmux; prints the URL
```

Everything heavy (OpenPI, the two venvs, the 12 GB checkpoint) installs under the persistent **`/workspace`** volume. `run.sh` launches both processes in **tmux**, so you can disconnect SSH and they keep running.

### Open the UI from your laptop

- **RunPod HTTP proxy (easiest):** `https://<POD_ID>-8888.proxy.runpod.net`
  (`echo $RUNPOD_POD_ID` on the pod, or see the pod's Connect panel).
- **SSH tunnel (lower latency, works even if 8888 isn't exposed):**
  ```bash
  ssh -L 8888:localhost:8888 root@<ip> -p <port> -i ~/.ssh/id_ed25519
  ```
  then open `http://localhost:8888`.

### Stop (to save GPU credits)

```bash
tmux kill-server     # stops the policy server + web UI, frees the GPU
```
Then stop the pod. The install persists on `/workspace`.

### Next time (fast restart)

- **Restarting the same pod, or attaching the same `/workspace` network volume to a new pod:** OpenPI + venvs + checkpoint are already there. Just re-run `./setup.sh` (fast — it only reinstalls the apt GL libs + tmux and re-seeds the LIBERO config, which live on the *ephemeral* container disk, then skips the big downloads) and `./run.sh`.
- **Brand-new pod without that volume:** full `./setup.sh` again (~15–20 min).

Notes: the first `run.sh` after a server (re)start waits ~5–8 min while JAX/OpenPI import off the network-FS venv and the checkpoint loads — it's in tmux, so just let it go. Re-running `run.sh` **reuses** an already-loaded server (only the web UI restarts); use `RESTART_SERVER=1 ./run.sh` to force a server reload. Logs: `tmux attach -t server` / `tmux attach -t webapp` (detach with `Ctrl-b d`).

---

## Using it

1. **Pick a scene** — choose a task suite (`libero_10` has the richest multi-object scenes) and a scene from the dropdown, then **Load scene & start**. The scene's canonical instruction is shown as a hint.
2. **Type an instruction** and hit **Send** (or Enter). The robot starts acting toward it.
3. **Watch** the live camera view (the model's-eye `agentview`).
4. **Steer it** — send a new instruction any time. It triggers an **immediate replan**, so use it for corrections or staged subgoals.

> ⚠️ Language alone isn't magic: the instruction only works if the **objects and placements actually exist in the chosen scene**. Compose instructions from objects you can see. "Set the table" only makes sense in a scene that has table-setting objects.

### Two interaction modes

- **Mode A — one long prompt:** type the whole goal once, e.g. `put both the cream cheese and the butter in the basket`, and let it run.
- **Mode B — staged subgoals:** send one subgoal at a time, waiting for each to (mostly) complete before the next, e.g. `open the top drawer` → `put the bowl in the top drawer` → `close the drawer`. This is the more diagnostic mode for testing whether pi0.5 is a good *subgoal executor*.

### Suggested test ladder

1. The scene's canonical instruction, unchanged (sanity check vs. the benchmark).
2. A paraphrase of it.
3. A slightly longer composed instruction.
4. A staged subgoal sequence (Mode B).
5. A mid-episode correction ("actually, put it in the drawer").

The question this answers first isn't benchmark success — it's: **can you steer pi0.5 by changing the typed instruction?**

---

## What gets logged

Every loaded scene starts a new run under `runs/`:

```
runs/2026-06-25_140312_libero_10_t0/
  instructions.txt   # every instruction you typed, with timestamp + step
  rollout.mp4        # video of the rollout (the model's-eye view)
  actions.npy        # the action chunks executed
  metadata.json      # suite, scene, canonical language, seed, OpenPI commit, success
```

Pull them to your laptop with `scp`/`rsync`, e.g.:
```bash
rsync -avz -e "ssh -p <port> -i ~/.ssh/id_ed25519" root@<ip>:/path/to/interactive-pi/runs/ ./runs/
```

---

## How it works (and what was changed from stock OpenPI)

OpenPI already ships a LIBERO example with a **policy-server / sim-client** split (`examples/libero/main.py` ↔ `scripts/serve_policy.py`). We reuse its exact observation preprocessing, action-chunk replanning, and websocket client. The **only** substantive change: the prompt sent to the policy (`main.py` hard-codes the benchmark's fixed `task_description`) becomes a **live, user-controlled variable** you can change at any time.

- `app/interactive_libero.py` — the interactive runner + Flask web UI. A single worker thread owns the MuJoCo env and steps pi0.5 against the current prompt; the browser polls the latest frame at ~10 fps and POSTs instruction changes.
- `setup.sh` — reproducible install (OpenPI pinned commit + LIBERO submodule, the two venvs, checkpoint prefetch).
- `run.sh` — starts the policy server, waits for the checkpoint to load, then launches the UI.

---

## Troubleshooting

- **Server "died" while waiting / OOM:** pi0.5 + MuJoCo rendering share the GPU. On ≤16 GB cards, lower JAX's preallocation: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 ./run.sh`. Check `/workspace/setup-logs/server.log`.
- **MuJoCo EGL errors:** the runner defaults to `MUJOCO_GL=egl`. If your pod lacks EGL, try `MUJOCO_GL=osmesa ./run.sh` (CPU rendering, slower).
- **Blank / frozen video over the RunPod proxy:** the UI polls frames (no long-lived stream), so this is usually a paused rollout — check the Status panel, or use the SSH tunnel instead of the proxy.
- **First start is slow:** the server downloads the ~pi05_libero checkpoint from `gs://openpi-assets` on first run (cached afterward). `setup.sh` prefetches it to avoid this.

## Cost hygiene

Stop the pod when you're not actively testing — GPU credits drain whether or not you're typing. The install lives on the `/workspace` volume, so a restart only needs `./run.sh` again.
