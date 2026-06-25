#!/usr/bin/env bash
# One-command setup for the interactive pi0.5 + LIBERO runner on a fresh GPU pod
# (tested on RunPod, Ubuntu 22.04, NVIDIA GPU, Python 3.11).
#
#   ./setup.sh
#
# It installs OpenPI + LIBERO into $OPENPI_DIR (default /workspace/openpi) with the
# two virtualenvs OpenPI's LIBERO example needs:
#   - server venv   (Python 3.11, jax[cuda12])  -> runs the pi0.5 policy server
#   - libero venv   (Python 3.8)                -> runs the simulator + our web UI
#
# Re-running is safe (idempotent-ish): existing clones/venvs are reused.
set -euo pipefail

# --- config (override via env) ---------------------------------------------
OPENPI_DIR="${OPENPI_DIR:-/workspace/openpi}"
OPENPI_COMMIT="${OPENPI_COMMIT:-15a9616}"   # pinned for reproducibility
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"   # keep wheels off the small root disk
PREFETCH_CHECKPOINT="${PREFETCH_CHECKPOINT:-1}"        # download pi05_libero now (set 0 to skip)
export UV_CACHE_DIR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CYAN='\033[36m'; NC='\033[0m'
step() { echo -e "\n${CYAN}=== $* ===${NC}"; }

# --- 0. system GL libraries (needed by MuJoCo EGL + robosuite) --------------
# RunPod base images ship the NVIDIA EGL vendor lib but not the GLVND loader
# (libEGL.so.1) or mesa GL, so robosuite/MuJoCo offscreen rendering fails without
# these. Safe to skip if you lack apt/root (set SKIP_APT=1).
if [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
  step "0/6  system GL libs (libegl1, libgl1, mesa)"
  apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libegl1 libgl1 libglib2.0-0 libosmesa6 libgl1-mesa-dri >/dev/null 2>&1 || \
    echo "  (apt install failed — install libegl1/libgl1 manually if rendering breaks)"
fi

# --- 1. uv -----------------------------------------------------------------
step "1/6  install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- 2. clone OpenPI + submodules ------------------------------------------
step "2/6  clone OpenPI (@ $OPENPI_COMMIT) + LIBERO submodule"
if [ ! -d "$OPENPI_DIR/.git" ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi
cd "$OPENPI_DIR"
git checkout -q "$OPENPI_COMMIT"
git submodule update --init --recursive
echo "OpenPI at $(git rev-parse --short HEAD)"

# --- 3. server venv (pi0.5 policy, jax[cuda12]) ----------------------------
step "3/6  server venv (Python 3.11 + jax[cuda12])"
GIT_LFS_SKIP_SMUDGE=1 uv sync

# --- 4. LIBERO client venv (Python 3.8) ------------------------------------
step "4/6  LIBERO client venv (Python 3.8)"
LV="$OPENPI_DIR/examples/libero/.venv"
uv venv --python 3.8 "$LV"
uv pip sync --python "$LV/bin/python" \
  examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match

step "5/6  editable installs (openpi-client, libero) + web deps"
uv pip install --python "$LV/bin/python" \
  -e packages/openpi-client -e third_party/libero flask \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match

# LIBERO's top-level package has no __init__.py (namespace pkg), so the PEP660
# editable finder doesn't expose it — OpenPI's own README adds it to PYTHONPATH.
export PYTHONPATH="$OPENPI_DIR/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
# LIBERO prompts interactively on first import to create ~/.libero/config.yaml;
# answer "N" once (use defaults) so it never blocks a non-interactive run.
printf 'N\n' | MUJOCO_GL=egl "$LV/bin/python" -c "from libero.libero import benchmark" >/dev/null 2>&1 || true
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$LV/bin/python" -c \
  "import libero, openpi_client, flask, cv2, mujoco, robosuite; from libero.libero import benchmark; print('client venv OK:', list(benchmark.get_benchmark_dict())[:3])"

# --- 6. (optional) pre-download the pi0.5 LIBERO checkpoint -----------------
if [ "$PREFETCH_CHECKPOINT" = "1" ]; then
  step "6/6  pre-download pi05_libero checkpoint (gs://openpi-assets, public)"
  uv run python -c "from openpi.shared import download; print('cached at:', download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
else
  step "6/6  skipping checkpoint prefetch (will download on first server start)"
fi

# --- record where things live for run.sh -----------------------------------
cat > "$REPO_DIR/.openpi_env" <<EOF
OPENPI_DIR=$OPENPI_DIR
LIBERO_VENV=$LV
LIBERO_PYTHONPATH=$OPENPI_DIR/third_party/libero
OPENPI_COMMIT=$(git -C "$OPENPI_DIR" rev-parse --short HEAD)
UV_CACHE_DIR=$UV_CACHE_DIR
EOF

echo -e "\n${CYAN}Setup complete.${NC} Start it with:  ./run.sh"
