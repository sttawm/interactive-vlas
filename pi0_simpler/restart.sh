#!/usr/bin/env bash
# Easy restart after a RunPod stop/start.
#
#   ./restart.sh
#
# The Volume Disk (/workspace) persists the heavy install — the CoVer clone, .venv_cover, the
# π0 + verifier checkpoints — but a stop/start WIPES the container disk (apt Vulkan/GL libs,
# tmux, uv, the sudo shim). So this re-runs setup.sh (idempotent: it repairs those and skips
# the big downloads / venv build, ~2-3 min) then run.sh. Runs in a detached tmux session, so
# you can close your laptop and it keeps booting.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${INNER:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null 2>&1 || true
  fi
  mkdir -p /workspace/setup-logs
  tmux kill-session -t boot 2>/dev/null || true
  tmux new-session -d -s boot "cd '$REPO_DIR' && INNER=1 ./restart.sh >/workspace/setup-logs/restart.log 2>&1"
  echo "Booting in tmux session 'boot' (~3-5 min). It survives SSH disconnects."
  echo "  watch:  tmux attach -t boot     (or: tail -f /workspace/setup-logs/restart.log)"
  echo "  the web UI URL is printed at the end of that log."
  exit 0
fi

cd "$REPO_DIR"
./setup.sh
VERIFIER="${VERIFIER:-0}" ./run.sh
