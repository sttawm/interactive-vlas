# Running interactive-vlas on RunPod

Everything RunPod-specific in one place. The per-instance `setup.sh` / `run.sh` /
`restart.sh` do the work; this explains how to create the pod and where things live.

## Storage model (the important part)

RunPod gives each pod two disks:

| | mount | persists? | speed | what we put there |
|---|---|---|---|---|
| **Container disk** | `/` | **wiped on stop/terminate** | fast (local) | apt packages, tmux, uv, `~/.libero` — all *restored* by setup/restart |
| **Volume disk** | `/workspace` | **survives stop/restart** (deleted on *terminate*) | slow (MooseFS/FUSE) | OpenPI/MolmoBot, venvs, checkpoints, repo, caches |

So **install everything to `/workspace`** (the per-instance `setup.sh` already does —
`OPENPI_DIR`, `UV_CACHE_DIR`, `OPENPI_DATA_HOME`, the HF cache all default there). A pod
**stop → start** keeps `/workspace` but wipes `/`, so coming back is a quick `./restart.sh`
(re-installs apt libs + reseeds config, skips the big downloads). A **terminate** deletes
`/workspace` → full `./setup.sh` again.

## Creating a pod

- **GPU ≥ 20 GB** (RTX 3090 / 4090 / A4500 / A5000). One VLA per pod — the two models
  can't co-host on one 24 GB card.
- **CUDA 12.x** image, Ubuntu 22.04, Python 3.11.
- **Volume disk ≥ 60 GB** mounted at `/workspace` (pi0.5 needs ~40 GB, MolmoBot ~50 GB;
  20–25 GB volumes hit `Disk quota exceeded`).
- Container disk ~20 GB is fine (nothing heavy lives there).
- **Expose port 8888 (HTTP)** and **22 (SSH)**.
- **Pick a region near you** — an EU pod viewed from the US is laggy (~3 fps live view)
  and slower to reach. (Headless evals don't care about region.)

## Bring-up

Fresh pod (one command bootstraps prereqs + clones to `/workspace`):

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/sttawm/interactive-vlas/main/runpod_bootstrap.sh)
```

Then pick an instance:

```bash
cd /workspace/interactive-vlas/pi05_libero          # or molmobot_molmospaces
./setup.sh        # one-time: install to /workspace (~15-20 min, download-bound)
./run.sh          # starts the server + UI in tmux; prints the URL
```

Open it at `https://<POD_ID>-8888.proxy.runpod.net` (`echo $RUNPOD_POD_ID`, or it's in the
pod's Connect panel). SSH-tunnel alt: `ssh -L 8888:localhost:8888 <pod>` → `http://localhost:8888`.

## Restart (after a stop/start)

```bash
cd /workspace/interactive-vlas/pi05_libero && ./restart.sh
```
One command, runs in detached tmux, survives disconnects. ~7–10 min for pi0.5 (mostly the
FUSE import + checkpoint load), ~6 min for MolmoBot (model load).

## Stop / cost

`tmux kill-server` frees the GPU; then **stop the pod** in the console to stop billing —
`/workspace` persists, so `./restart.sh` brings it back. **Terminate** only when you're
done for good (it deletes `/workspace`).

## Known caveats

- First server start is ~5–8 min: importing JAX/torch/openpi off the FUSE volume is
  open-latency bound. Inference is fast once loaded. (Call the venv python directly, not
  `uv run`, which revalidates the whole env — `run.sh` already does this.)
- SSH can be flaky (drops/`exit 255`): use `-o ServerAliveInterval=5`, keep commands
  short, run long things under tmux.
- A shared **network volume** (in one region, attached per running pod) would let new
  pods skip re-downloading checkpoints — useful future optimization; doesn't remove the
  per-pod venv rebuild or FUSE import wait.
