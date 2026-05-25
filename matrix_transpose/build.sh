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
hipcc -std=c++17 -O3 -g --offload-arch="${ARCH}" naive_transpose.cpp -o naive_transpose
hipcc -std=c++17 -O3 -g --offload-arch="${ARCH}" optimized_transpose.cpp -o optimized_transpose
echo "Built: naive_transpose (HIP_ARCH=${ARCH})"
echo "Built: optimized_transpose (HIP_ARCH=${ARCH})"
