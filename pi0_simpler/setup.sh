#!/usr/bin/env bash
# One-command setup for the interactive π0-Bridge + SimplerEnv runner on a fresh GPU pod
# (target: RunPod, Ubuntu 22.04, NVIDIA GPU, Python 3.10, CUDA 12.x).
#
#   ./setup.sh
#
# Like molmobot_molmospaces (and unlike pi05_libero) there is NO separate policy server: the
# π0 policy (LeRobot PI0Policy) runs in-process with the SimplerEnv (SAPIEN/ManiSkill2) sim.
#
# It clones the CoVer repo (github.com/cover-vla/cover-vla) into $COVER_DIR, runs the repo's
# own env_simpler_pi.sh to build the .venv_cover (Python 3.10: TF + torch cu128 + SimplerEnv +
# lerobot_custom[pi0] + bridge_verifier), adds the web-UI deps, and prefetches the checkpoint(s).
# Re-running is safe.
#
# ⚠️  SimplerEnv WidowX rendering uses SAPIEN → needs a working **Vulkan** ICD (NVIDIA), not
#     just EGL/OSMesa. This is the usual SimplerEnv-on-a-pod snag; see the Vulkan check below
#     and README "Troubleshooting".
set -euo pipefail

# --- config (override via env) ---------------------------------------------
COVER_DIR="${COVER_DIR:-/workspace/cover-vla}"
COVER_COMMIT="${COVER_COMMIT:-00af094}"                      # pinned for reproducibility
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"          # checkpoint cache on persistent vol
CHECKPOINT="${CHECKPOINT:-juexzz/INTACT-pi0-finetune-rephrase-bridge}"
PREFETCH_CHECKPOINT="${PREFETCH_CHECKPOINT:-1}"             # download π0 ckpt now (0 to skip)
PREFETCH_VERIFIER="${PREFETCH_VERIFIER:-1}"                 # download CoVer verifier ckpt (0 to skip)
export UV_CACHE_DIR HF_HOME

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$COVER_DIR/.venv_cover"
INFERENCE_ROOT="$COVER_DIR/CoVer_VLA/inference"
CYAN='\033[36m'; NC='\033[0m'
step() { echo -e "\n${CYAN}=== $* ===${NC}"; }

# --- 0. system deps: GL + Vulkan (SAPIEN offscreen render) + tmux -----------
if [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
  step "0/6  system deps (Vulkan + GL libs for SAPIEN/ManiSkill2 + tmux)"
  ${SUDO:-} apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive ${SUDO:-} apt-get install -y -qq \
    libvulkan1 libx11-6 vulkan-tools libegl1 libgl1 libglib2.0-0 libosmesa6 libgl1-mesa-dri tmux \
    >/dev/null 2>&1 || echo "  (apt install failed — install libvulkan1/libx11-6/tmux manually if needed)"
fi

# --- 0a. Vulkan preflight (fail fast before the ~20 min build) --------------
# SimplerEnv/SAPIEN renders through Vulkan; a pod without it can't run this viewer at all.
if [ "${SKIP_VULKAN_CHECK:-0}" != "1" ]; then
  step "0a  Vulkan preflight (SimplerEnv/SAPIEN requires it)"
  if bash "$REPO_DIR/../tools/check_vulkan.sh"; then
    :
  else
    echo ""
    echo "  ⚠️  Vulkan is not available on this pod (see the fix above)."
    echo "     The build can continue, but rendering WILL fail until Vulkan works."
    echo "     Set VULKAN_REQUIRED=1 to abort here instead; SKIP_VULKAN_CHECK=1 to skip this check."
    [ "${VULKAN_REQUIRED:-0}" = "1" ] && { echo "  Aborting (VULKAN_REQUIRED=1)."; exit 1; }
  fi
fi

# CoVer's env_simpler_pi.sh calls `sudo apt-get`; on a root pod without sudo that aborts (set -e).
# Provide a passthrough `sudo` shim so it just runs the command as-is.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v sudo >/dev/null 2>&1; then
  step "0b  no sudo found — installing a passthrough sudo shim (root container)"
  mkdir -p "$HOME/.local/bin"
  printf '#!/usr/bin/env bash\nexec "$@"\n' > "$HOME/.local/bin/sudo"
  chmod +x "$HOME/.local/bin/sudo"
fi

# --- 1. uv -----------------------------------------------------------------
step "1/6  install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- 2. clone CoVer --------------------------------------------------------
step "2/6  clone CoVer (@ $COVER_COMMIT)"
if [ ! -d "$COVER_DIR/.git" ]; then
  git clone https://github.com/cover-vla/cover-vla.git "$COVER_DIR"
fi
git -C "$COVER_DIR" checkout -q "$COVER_COMMIT" || echo "  (could not checkout $COVER_COMMIT — using current HEAD)"
echo "CoVer at $(git -C "$COVER_DIR" rev-parse --short HEAD)"

# --- 3. build the CoVer venv via their own script --------------------------
# Builds $COVER_DIR/.venv_cover (Python 3.10) with SimplerEnv + lerobot[pi0] + bridge_verifier.
step "3/6  build .venv_cover (CoVer_VLA/scripts/env_simpler_pi.sh — several minutes)"
if [ ! -x "$VENV/bin/python" ]; then
  ( cd "$COVER_DIR" && bash CoVer_VLA/scripts/env_simpler_pi.sh )
else
  echo "  .venv_cover already exists — skipping (delete it to rebuild)."
fi

# --- 4. web-UI deps into the CoVer venv ------------------------------------
step "4/6  web deps into .venv_cover (flask/opencv/imageio/anthropic)"
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
  flask opencv-python-headless imageio imageio-ffmpeg anthropic

# --- 5. sanity imports (policy + sim + web) --------------------------------
step "5/6  sanity import (PI0Policy + simpler_env + bridge_verifier + web deps)"
PYTHONPATH="$COVER_DIR:$INFERENCE_ROOT" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  "$VENV/bin/python" -c "
import flask, cv2, imageio
import simpler_env
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from experiments.robot.simpler.eval_utils import create_bridge_adapter_wrapper
print('imports OK — simpler_env, PI0Policy, adapter, web deps')
" || { echo '✗ sanity import failed — check the venv build log above.'; exit 1; }

# Vulkan smoke test: build one WidowX scene + render a frame (the real SimplerEnv gate).
echo "  Vulkan/render smoke test (simpler_env.make + reset + render one frame)…"
PYTHONPATH="$COVER_DIR:$INFERENCE_ROOT" \
  "$VENV/bin/python" -c "
import numpy as np, simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
env = simpler_env.make('widowx_spoon_on_towel'); obs,_ = env.reset(seed=0)
img = get_image_from_maniskill2_obs_dict(env, obs)
print('rendered frame', np.asarray(img).shape, np.asarray(img).dtype)
" || echo "  ⚠️  render smoke test FAILED — SAPIEN/Vulkan not working on this pod (see README Troubleshooting)."

# --- 6. prefetch checkpoints -----------------------------------------------
step "6/6  prefetch checkpoints"
if [ "$PREFETCH_CHECKPOINT" = "1" ]; then
  echo "  π0 policy: $CHECKPOINT"
  HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
print('cached at:', snapshot_download('$CHECKPOINT'))
"
fi
if [ "$PREFETCH_VERIFIER" = "1" ]; then
  echo "  CoVer verifier: cover-vla/cover-vla-bridge/cover_verifier_bridge.pt"
  "$VENV/bin/python" -c "
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download('cover-vla/cover-vla-bridge', 'cover_verifier_bridge.pt')
dst = os.path.join('$COVER_DIR','bridge_verifier','cover_verifier_bridge.pt')
if os.path.abspath(p) != os.path.abspath(dst): shutil.copy(p, dst)
print('verifier at:', dst)
" || echo "  (verifier download failed — only needed for VERIFIER=1 runs)"
fi

# --- record paths for run.sh -----------------------------------------------
cat > "$REPO_DIR/.cover_env" <<EOF
COVER_DIR=$COVER_DIR
COVER_VENV=$VENV
COVER_INFERENCE=$INFERENCE_ROOT
COVER_PYTHONPATH=$COVER_DIR:$INFERENCE_ROOT
COVER_COMMIT=$(git -C "$COVER_DIR" rev-parse --short HEAD)
CHECKPOINT=$CHECKPOINT
UV_CACHE_DIR=$UV_CACHE_DIR
HF_HOME=$HF_HOME
EOF

echo -e "\n${CYAN}Setup complete.${NC} Start it with:  ./run.sh   (verifier: VERIFIER=1 ./run.sh)"
