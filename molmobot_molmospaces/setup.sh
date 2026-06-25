#!/usr/bin/env bash
# One-command setup for the interactive MolmoBot + MolmoSpaces runner on a fresh GPU
# pod (tested target: RunPod, Ubuntu 22.04, NVIDIA GPU, Python 3.11).
#
#   ./setup.sh
#
# Unlike pi05_libero there is NO separate policy server: the MolmoBot VLA runs
# in-process with the MuJoCo sim. So this just needs one environment.
#
# It clones MolmoBot into $MOLMOBOT_DIR (default /workspace/MolmoBot) and installs it
# with the `eval` extra, which transitively pulls a pinned molmo_spaces[mujoco]. Then
# it adds the web deps (flask/opencv/imageio) and pre-downloads the MolmoBot-DROID
# checkpoint. Re-running is safe.
set -euo pipefail

# --- config (override via env) ---------------------------------------------
MOLMOBOT_DIR="${MOLMOBOT_DIR:-/workspace/MolmoBot}"
HF_REPO="${HF_REPO:-allenai/MolmoBot-DROID}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"        # checkpoint cache on persistent vol
MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-/workspace/.cache/molmospaces}"  # scene assets
PREFETCH_CHECKPOINT="${PREFETCH_CHECKPOINT:-1}"            # download MolmoBot-DROID now (0 to skip)
export UV_CACHE_DIR HF_HOME MLSPACES_ASSETS_DIR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$MOLMOBOT_DIR/MolmoBot"   # the python package + pyproject live one level down
CYAN='\033[36m'; NC='\033[0m'
step() { echo -e "\n${CYAN}=== $* ===${NC}"; }

# --- 0. system GL libraries (MuJoCo EGL offscreen render) -------------------
if [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
  step "0/5  system GL libs (libegl1, libgl1, mesa)"
  apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libegl1 libgl1 libglib2.0-0 libosmesa6 libgl1-mesa-dri >/dev/null 2>&1 || \
    echo "  (apt install failed — install libegl1/libgl1 manually if rendering breaks)"
fi

# --- 1. uv -----------------------------------------------------------------
step "1/5  install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- 2. clone MolmoBot -----------------------------------------------------
step "2/5  clone MolmoBot"
if [ ! -d "$MOLMOBOT_DIR/.git" ]; then
  git clone https://github.com/allenai/MolmoBot.git "$MOLMOBOT_DIR"
fi
echo "MolmoBot at $(git -C "$MOLMOBOT_DIR" rev-parse --short HEAD)"

# --- 3. install MolmoBot + eval extra (pulls molmo_spaces[mujoco]) ----------
step "3/5  uv sync --extra eval (this also installs the pinned molmo_spaces)"
cd "$PKG_DIR"
uv sync --extra eval
# web UI deps not in MolmoBot's pyproject
uv pip install flask opencv-python-headless imageio imageio-ffmpeg

# --- 4. sanity import -------------------------------------------------------
step "4/5  sanity import (molmo_spaces scene API + olmo policy + web deps)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PKG_DIR/.venv/bin/python" -c "
import flask, cv2, imageio, mujoco
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses
from olmo.eval.configure_real_robot import RealRobotVLAPolicy
print('env OK — molmo_spaces + olmo + web deps import')
"

# --- 5. (optional) pre-download the checkpoint ------------------------------
if [ "$PREFETCH_CHECKPOINT" = "1" ]; then
  step "5/5  pre-download $HF_REPO (~8-9 GB)"
  HF_HUB_ENABLE_HF_TRANSFER=1 "$PKG_DIR/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
print('cached at:', snapshot_download('$HF_REPO'))
"
else
  step "5/5  skipping checkpoint prefetch (downloads on first run)"
fi

# --- record paths for run.sh -----------------------------------------------
cat > "$REPO_DIR/.molmobot_env" <<EOF
MOLMOBOT_DIR=$MOLMOBOT_DIR
MOLMOBOT_VENV=$PKG_DIR/.venv
HF_REPO=$HF_REPO
UV_CACHE_DIR=$UV_CACHE_DIR
HF_HOME=$HF_HOME
MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR
MOLMOBOT_COMMIT=$(git -C "$MOLMOBOT_DIR" rev-parse --short HEAD)
EOF

echo -e "\n${CYAN}Setup complete.${NC} Start it with:  ./run.sh"
