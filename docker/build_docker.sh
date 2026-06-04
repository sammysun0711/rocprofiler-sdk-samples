#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.gfx950-rocm7.13-vllm0.19"
#IMAGE_NAME="rocm/vllm-dev:vllm-0.19.0-ubuntu22.04-py3.12-rocm7.13-0414-gfx950"
IMAGE_NAME="sammysun0711/vllm-dev:vllm-0.19.0-ubuntu22.04-py3.12-rocm7.13-0414-gfx950"
LOG_FILE="${SCRIPT_DIR}/docker_build.log"

echo "Building image: ${IMAGE_NAME}"
echo "Dockerfile: ${DOCKERFILE}"
echo "Log file: ${LOG_FILE}"

docker buildx build \
    --progress=plain \
    -t "${IMAGE_NAME}" \
    -f "${DOCKERFILE}" \
    "${SCRIPT_DIR}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Build complete. Log saved to: ${LOG_FILE}"
