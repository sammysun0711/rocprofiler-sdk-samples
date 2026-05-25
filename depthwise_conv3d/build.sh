#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${HIP_ARCH:-}" ] && command -v /opt/rocm/bin/rocminfo &> /dev/null; then
    ARCH=$(/opt/rocm/bin/rocminfo 2>/dev/null | grep "Name:" | grep -oP "gfx\w+" | head -1)
    ARCH="${ARCH:-gfx942}"
else
    ARCH="${HIP_ARCH:-gfx942}"
fi
cd "$ROOT"
hipcc -std=c++17 -O3 -g --offload-arch="${ARCH}" -DKERNEL_SGB=0 bench_conv3d.cpp -o conv3d_depthwise_original
hipcc -std=c++17 -O3 -g --offload-arch="${ARCH}" -DKERNEL_SGB=1 bench_conv3d.cpp -o conv3d_depthwise_sgb
echo "Built: conv3d_depthwise_original finished  (HIP_ARCH=${ARCH})"
echo "Built: conv3d_depthwise_sgb finished (HIP_ARCH=${ARCH})"
