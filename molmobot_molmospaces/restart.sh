#!/usr/bin/env bash
# Easy restart after a RunPod stop/start (MolmoBot + MolmoSpaces).
#
#   ./restart.sh
#
# The Volume Disk (/workspace) persists the heavy install — MolmoBot, the venv, the
# checkpoint, and the scene assets — but a stop/start WIPES the container disk (apt GL
# libs, tmux). So this re-runs setup.sh (idempotent: repairs those, skips downloads,
# ~2-3 min), then launches the in-process app (model loads ~3 min) in a detached 'mb'
# tmux session. Survives SSH disconnects.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Re-exec inside a detached tmux 'boot' session (install tmux first — a restart wiped it).
if [ "${INNER:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null 2>&1 || true
  fi
  mkdir -p /workspace/setup-logs
  tmux kill-session -t boot 2>/dev/null || true
  tmux new-session -d -s boot "cd '$REPO_DIR' && INNER=1 ./restart.sh >/workspace/setup-logs/restart.log 2>&1"
  echo "Booting in tmux session 'boot' (~6 min: setup repair + model load). Survives disconnects."
  echo "  watch:  tail -f /workspace/setup-logs/restart.log   (then: tmux attach -t mb)"
  exit 0
fi

cd "$REPO_DIR"
./setup.sh
# Launch the in-process app in its own 'mb' tmux session (model loads ~3 min).
tmux kill-session -t mb 2>/dev/null || true
tmux new-session -d -s mb "cd '$REPO_DIR' && ./run.sh >/workspace/setup-logs/run.log 2>&1"
echo "App starting in tmux 'mb' (model load ~3 min). Watch: tail -f /workspace/setup-logs/run.log"
