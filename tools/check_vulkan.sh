#!/usr/bin/env bash
# Quick go/no-go: can this pod render with Vulkan? SimplerEnv/SAPIEN (pi0_simpler) needs it —
# and it's the #1 reason a GPU pod fails. Run this BEFORE the ~20 min setup to fail fast.
#
#   bash tools/check_vulkan.sh
#
# Prints PASS with the GPU name, or FAIL with the most common RunPod fix. Exit 0 = ok, 1 = no Vulkan.
set -uo pipefail
SUDO="${SUDO:-}"

if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "Installing vulkan loader + tools (one-time)…"
  ${SUDO} apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get install -y -qq libvulkan1 vulkan-tools >/dev/null 2>&1 || true
fi

echo "== NVIDIA driver (nvidia-smi — necessary but NOT sufficient for Vulkan) =="
nvidia-smi -L 2>/dev/null | sed 's/^/  /' || echo "  nvidia-smi not found — no NVIDIA driver visible."

echo "== NVIDIA Vulkan ICD present? (needs the 'graphics' driver capability) =="
if ls /usr/share/vulkan/icd.d/*nvidia*.json >/dev/null 2>&1; then
  ls -1 /usr/share/vulkan/icd.d/*nvidia*.json | sed 's/^/  /'
else
  echo "  ⚠️  no nvidia_icd.json under /usr/share/vulkan/icd.d/ — Vulkan can't find the NVIDIA driver."
fi

echo "== vulkaninfo =="
out="$(vulkaninfo --summary 2>&1)"
dev="$(printf '%s\n' "$out" | grep -i 'deviceName' | head -1 | sed 's/.*=[[:space:]]*//')"
if [ -n "$dev" ]; then
  echo "✅ PASS — Vulkan sees a GPU: $dev"
  echo "   SimplerEnv/SAPIEN should render on this pod. Proceed with ./pi0_simpler/setup.sh"
  exit 0
fi

printf '%s\n' "$out" | tail -12 | sed 's/^/  /'
cat <<'EOF'

❌ FAIL — no Vulkan device found. SimplerEnv/SAPIEN will NOT render here.
   nvidia-smi / CUDA can work fine while Vulkan does not — they're separate.

   Most common RunPod cause: the container lacks the 'graphics' driver capability.
   Fix: recreate the pod with the environment variable
        NVIDIA_DRIVER_CAPABILITIES=all        (or: compute,utility,graphics)

   If the driver ships its ICD at a nonstandard path, point the loader at it:
        export VK_ICD_FILENAMES=/path/to/nvidia_icd.json
   (find it with:  find / -name 'nvidia_icd*.json' 2>/dev/null )
EOF
exit 1
