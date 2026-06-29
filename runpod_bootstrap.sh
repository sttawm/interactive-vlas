#!/usr/bin/env bash
# Fresh-RunPod bootstrap for interactive-vlas. Run FIRST on a brand-new pod:
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/sttawm/interactive-vlas/main/runpod_bootstrap.sh)
#
# Installs the apt prereqs RunPod base images lack (tmux, git-lfs — both live on the
# ephemeral container disk, so they're gone after every stop), clones the repo onto the
# persistent /workspace volume, and prints the next step. Safe to re-run. See RUNPOD.md.
set -euo pipefail
REPO_URL="https://github.com/sttawm/interactive-vlas.git"
WS="${WS:-/workspace}"

echo "== interactive-vlas RunPod bootstrap =="
if command -v apt-get >/dev/null 2>&1; then
  echo "Installing prereqs (tmux, git-lfs)..."
  apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux git-lfs >/dev/null 2>&1 || true
  git lfs install >/dev/null 2>&1 || true
fi

mkdir -p "$WS/setup-logs"
cd "$WS"
[ -d interactive-vlas/.git ] || git clone -q "$REPO_URL"
git -C "$WS/interactive-vlas" pull -q 2>/dev/null || true
echo "Repo ready: $WS/interactive-vlas ($(git -C "$WS/interactive-vlas" rev-parse --short HEAD))"

cat <<EOF

Next — pick an instance, then setup (one-time) or restart (after a pod stop):
  pi0.5+LIBERO :  cd $WS/interactive-vlas/pi05_libero          && ./setup.sh && ./run.sh
  MolmoBot     :  cd $WS/interactive-vlas/molmobot_molmospaces && ./setup.sh && ./run.sh
  after a stop :  cd <instance> && ./restart.sh
Open: https://\${RUNPOD_POD_ID}-8888.proxy.runpod.net    (full guide: RUNPOD.md)
EOF
