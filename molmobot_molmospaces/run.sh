#!/usr/bin/env bash
# Launch the interactive MolmoBot + MolmoSpaces web UI (model runs in-process; no
# separate policy server).
#
#   ./run.sh                       # MolmoBot-DROID, web UI on :8888
#   HOUSE=3 ./run.sh               # start on a different ProcTHOR-10k house
#   STUB=1 ./run.sh                # no-model wobble policy (plumbing test, no GPU)
#
# Open the web UI from your laptop via the RunPod HTTP proxy:
#   https://<POD_ID>-8888.proxy.runpod.net
# or an SSH tunnel:  ssh -L 8888:localhost:8888 root@<ip> -p <port>  then http://localhost:8888
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO_DIR/.molmobot_env" ] || { echo "Missing .molmobot_env — run ./setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source "$REPO_DIR/.molmobot_env"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR HF_HOME MLSPACES_ASSETS_DIR
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

WEB_PORT="${WEB_PORT:-8888}"
HOUSE="${HOUSE:-0}"
SPLIT="${SPLIT:-val}"
EXTRA=()
[ "${STUB:-0}" = "1" ] && EXTRA+=(--stub)
[ -n "${CHECKPOINT:-}" ] && EXTRA+=(--checkpoint "$CHECKPOINT")

echo "Starting interactive web UI on :$WEB_PORT  (house $HOUSE/$SPLIT, repo ${HF_REPO})"
exec "$MOLMOBOT_VENV/bin/python" "$REPO_DIR/app/interactive_molmospaces.py" \
  --web-port "$WEB_PORT" --house-index "$HOUSE" --split "$SPLIT" \
  --checkpoint "${CHECKPOINT:-$HF_REPO}" "${EXTRA[@]}"
