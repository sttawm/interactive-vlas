#!/usr/bin/env bash
# Launch the interactive π0-Bridge + SimplerEnv web UI (policy runs in-process; no separate
# policy server). Runs in a tmux session so it survives SSH disconnects.
#
#   ./run.sh                       # π0 (rephrase), plain live-prompt, web UI on :8888
#   VERIFIER=1 ./run.sh            # run the CoVer verifier loop (needs the verifier ckpt)
#   CHECKPOINT=juexzz/INTACT-pi0-finetune-bridge ./run.sh   # no-paraphrase baseline policy
#   STUB=1 ./run.sh                # no policy/sim — animated placeholder (CPU plumbing test)
#
# Open the web UI from your laptop:
#   - RunPod proxy:  https://<POD_ID>-8888.proxy.runpod.net   (POD_ID = $RUNPOD_POD_ID)
#   - or SSH tunnel: ssh -L 8888:localhost:8888 <ssh-to-pod>  then http://localhost:8888
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO_DIR/.cover_env" ] || { echo "Missing .cover_env — run ./setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source "$REPO_DIR/.cover_env"
export PATH="$HOME/.local/bin:$PATH"

WEB_PORT="${WEB_PORT:-8888}"
# SAPIEN uses Vulkan for rendering; osmesa/egl are the fallbacks other GL bits use.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
# Expose CoVer paths to the app (rephrase JSON, verifier ckpt) + import roots.
export PYTHONPATH="$COVER_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export COVER_DIR COVER_INFERENCE COVER_COMMIT HF_HOME

CKPT="${CHECKPOINT:-juexzz/INTACT-pi0-finetune-rephrase-bridge}"
EXTRA=()
[ "${VERIFIER:-0}" = "1" ] && EXTRA+=(--verifier)
[ "${STUB:-0}" = "1" ] && EXTRA+=(--stub)
[ -n "${LANG_REPHRASE_NUM:-}" ] && EXTRA+=(--lang-rephrase-num "$LANG_REPHRASE_NUM")
[ -n "${BATCH:-}" ] && EXTRA+=(--policy-batch-inference-size "$BATCH")
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-rollout-steps "$MAX_STEPS")

command -v tmux >/dev/null 2>&1 || { echo "Installing tmux…"; ${SUDO:-} apt-get install -y -qq tmux >/dev/null 2>&1 || true; }
LOGDIR=/workspace/setup-logs; mkdir -p "$LOGDIR"
port_open() { (echo >/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }

echo "Starting interactive web UI in tmux on :$WEB_PORT  (ckpt=$CKPT, verifier=${VERIFIER:-0}, stub=${STUB:-0})"
tmux kill-session -t webapp 2>/dev/null || true
: > "$LOGDIR/webapp.log"
tmux new-session -d -s webapp \
  "cd '$REPO_DIR' && PYTHONUNBUFFERED=1 MUJOCO_GL='$MUJOCO_GL' PYOPENGL_PLATFORM='$PYOPENGL_PLATFORM' \
   PYTHONPATH='$PYTHONPATH' COVER_DIR='$COVER_DIR' COVER_INFERENCE='$COVER_INFERENCE' \
   COVER_COMMIT='$COVER_COMMIT' HF_HOME='$HF_HOME' \
   '$COVER_VENV/bin/python' -u app/interactive_simpler.py \
     --web-port $WEB_PORT --checkpoint '$CKPT' ${EXTRA[*]} 2>&1 | tee '$LOGDIR/webapp.log'"

echo "Waiting for the web UI (first start loads π0 + builds the SAPIEN scene; can take a few min)…"
for i in $(seq 1 90); do
  port_open "$WEB_PORT" && break
  tmux has-session -t webapp 2>/dev/null || { echo "✗ webapp tmux session died. Last log:"; tail -30 "$LOGDIR/webapp.log"; exit 1; }
  sleep 2
done
port_open "$WEB_PORT" && echo "✓ Web UI is up." || echo "… still starting (check: tmux attach -t webapp)"

echo
POD_ID="${RUNPOD_POD_ID:-}"
[ -z "$POD_ID" ] && POD_ID="$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p')"
echo "──────────────────────────────────────────────────────────────"
if [ -n "$POD_ID" ]; then
  echo "  Open:  https://${POD_ID}-${WEB_PORT}.proxy.runpod.net"
else
  echo "  Open:  https://<POD_ID>-${WEB_PORT}.proxy.runpod.net   (set POD_ID from the RunPod Connect panel)"
fi
echo "  Tunnel alt: ssh -L ${WEB_PORT}:localhost:${WEB_PORT} <ssh-to-pod>  then http://localhost:${WEB_PORT}"
echo "  Logs:  tmux attach -t webapp   (detach: Ctrl-b d)"
echo "  Stop:  tmux kill-session -t webapp   (frees the GPU)"
echo "──────────────────────────────────────────────────────────────"
