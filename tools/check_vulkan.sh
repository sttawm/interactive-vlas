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
# Require a HARDWARE GPU. Reject the CPU software fallback (llvmpipe / lavapipe / swiftshader):
# it makes vulkaninfo "succeed" but SAPIEN would run on CPU (unusably slow) or crash on missing
# features. A real device has deviceType != PHYSICAL_DEVICE_TYPE_CPU.
hw_gpu="$(printf '%s\n' "$out" | grep -i 'deviceName' | grep -viE 'llvmpipe|lavapipe|swiftshader|software' | head -1 | sed 's/.*=[[:space:]]*//')"
has_hw="$(printf '%s\n' "$out" | grep -i 'deviceType' | grep -viE 'CPU' | head -1)"
if [ -n "$hw_gpu" ] && [ -n "$has_hw" ]; then
  echo "✅ PASS — Vulkan sees a hardware GPU: $hw_gpu"
  echo "   SimplerEnv/SAPIEN should render on this pod. Proceed with ./pi0_simpler/setup.sh"
  exit 0
fi
if printf '%s\n' "$out" | grep -qiE 'llvmpipe|lavapipe|swiftshader'; then
  echo "  ⚠️  Only a CPU software renderer (llvmpipe) is available — NOT the GPU."
fi

printf '%s\n' "$out" | tail -6 | sed 's/^/  /'
cat <<'EOF'

❌ FAIL — no *hardware* Vulkan device. SimplerEnv/SAPIEN will NOT render usably here.
   nvidia-smi / CUDA can work fine while GPU Vulkan does not — they're separate stacks.

   Fixes, in order of likelihood on RunPod:
   1) Missing 'graphics' capability (no nvidia_icd.json under /etc/vulkan/icd.d or
      /usr/share/vulkan/icd.d): recreate the pod with the environment variable
        NVIDIA_DRIVER_CAPABILITIES=all        (or: compute,utility,graphics)
   2) Driver refuses the Vulkan instance ("vkCreateInstance: Found no drivers!") even though
      the NVIDIA libs/ICD are present → the GPU's graphics engine isn't exposed to this
      container. This is a HOST-level config; it usually can't be fixed from inside the pod.
      Terminate and redeploy on a different host/template, then re-run this check FIRST.
   3) Nonstandard ICD path: export VK_ICD_FILENAMES=/path/to/nvidia_icd.json
      (find it:  find / -name 'nvidia_icd*.json' 2>/dev/null )
EOF
exit 1
