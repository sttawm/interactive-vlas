#!/usr/bin/env bash
# Extension chain for a pod whose MAIN four-tier shard has completed:
#   E1 deepen  (same shard: orig 10->30, tiers 5->10/phrase, confirm2 on inits 30-49)
#   E2 coverage (shard <name>x: full four-tier on the extension libero_90 tasks)
# Appends to the SAME /workspace/fourtier_<shard>.jsonl; every cell resume-safe.
#
#   ./run_fourtier_ext.sh lb1
set -euo pipefail
SHARD="${1:?usage: run_fourtier_ext.sh <lb1|lb2|lb3|lb4>}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/.openpi_env"
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=/workspace/setup-logs
port_open() { (echo >/dev/tcp/127.0.0.1/"$1") 2>/dev/null; }
port_open 8000 || { echo "policy server not up — run run_fourtier.sh first"; exit 1; }

OUT="/workspace/fourtier_${SHARD}.jsonl"
LOG="/workspace/fourtier_${SHARD}.log"
tmux kill-session -t fourtier 2>/dev/null || true
tmux new-session -d -s fourtier \
  "cd '$REPO_DIR/eval' && PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
   '$LIBERO_VENV/bin/python' -u fourtier_eval.py --shard $SHARD --phase deepen --out $OUT 2>&1 | tee -a '$LOG' && \
   PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
   '$LIBERO_VENV/bin/python' -u fourtier_eval.py --shard ${SHARD}x --out $OUT 2>&1 | tee -a '$LOG' && \
   PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
   '$LIBERO_VENV/bin/python' -u fourtier_eval.py --shard $SHARD --phase oracleplus --out $OUT 2>&1 | tee -a '$LOG' && \
   PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
   '$LIBERO_VENV/bin/python' -u fourtier_eval.py --shard ${SHARD}x --phase oracleplus --out $OUT 2>&1 | tee -a '$LOG' && \
   PYTHONPATH='$LIBERO_PYTHONPATH' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
   '$LIBERO_VENV/bin/python' -u fourtier_eval.py --shard $SHARD --phase cross --out $OUT 2>&1 | tee -a '$LOG'; \
   echo \"ext chain exited rc=\$?\"; sleep infinity"
echo "EXT LAUNCHED shard=$SHARD (deepen -> ${SHARD}x coverage -> oracleplus both)"
