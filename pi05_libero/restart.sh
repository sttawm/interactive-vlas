#!/usr/bin/env bash
# Easy restart after a RunPod stop/start.
#
#   ./restart.sh
#
# The Volume Disk (/workspace) persists the heavy install — openpi, both venvs, the
# 12GB checkpoint — but a stop/start WIPES the container disk (apt GL libs, tmux,
# ~/.libero config, uv). So this re-runs setup.sh (idempotent: it repairs those and
# skips the big downloads, ~2-3 min) then run.sh (server load ~5-8 min off the
# network-FS venv). The whole thing runs in a detached tmux session, so you can close
# your laptop and it keeps booting.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Re-exec inside a detached tmux 'boot' session (a restart wiped tmux, so install it
# first) unless we're already the inner run.
if [ "${INNER:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null 2>&1 || true
  fi
  mkdir -p /workspace/setup-logs
  tmux kill-session -t boot 2>/dev/null || true
  tmux new-session -d -s boot "cd '$REPO_DIR' && INNER=1 ./restart.sh >/workspace/setup-logs/restart.log 2>&1"
  echo "Booting in tmux session 'boot' (~8-10 min). It survives SSH disconnects."
  echo "  watch:  tmux attach -t boot     (or: tail -f /workspace/setup-logs/restart.log)"
  echo "  the web UI URL is printed at the end of that log."
  exit 0
fi

cd "$REPO_DIR"
./setup.sh
./run.sh
