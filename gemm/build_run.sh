#!/bin/bash

# Build and run script for FP8 GEMM implementations
# Builds both naive and optimized versions in release and debug modes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Compiler and flags
CXX=hipcc

# Try to auto-detect GPU architecture, otherwise use default
if command -v /opt/rocm/bin/rocminfo &> /dev/null; then
    GPU_ARCH=$(/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 "Name:" | grep -oP "gfx\w+" || echo "gfx942")
else
    GPU_ARCH="gfx942"  # Default to MI300
fi

# Allow override via environment variable
GPU_ARCH=${GPU_ARCH:-gfx942}
ARCH="--offload-arch=${GPU_ARCH}"

# Common flags
COMMON_FLAGS="-std=c++17 -Wall -Wextra"

# Release flags
RELEASE_FLAGS="-O3"

# Debug flags
DEBUG_FLAGS="-g"

# Source files
NAIVE_SRC="naive_gemm.cpp"
OPTIMIZED_SRC="optimized_gemm.cpp"

# Output binaries
NAIVE_RELEASE="naive_gemm_release"
NAIVE_DEBUG="naive_gemm_debug"
OPTIMIZED_RELEASE="optimized_gemm_release"
OPTIMIZED_DEBUG="optimized_gemm_debug"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  FP8 GEMM Build Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU Architecture: ${GPU_ARCH}${NC}"
echo ""

# Function to build a target
build_target() {
    local source=$1
    local output=$2
    local flags=$3
    local build_type=$4
    
    echo -e "${YELLOW}Building ${output} (${build_type})...${NC}"
    
    if ${CXX} ${ARCH} ${COMMON_FLAGS} ${flags} ${source} -o ${output}; then
        echo -e "${GREEN}✓ Successfully built ${output}${NC}"
        echo ""
    else
        echo -e "${RED}✗ Failed to build ${output}${NC}"
        exit 1
    fi
}

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -f ${NAIVE_RELEASE} ${NAIVE_DEBUG} ${OPTIMIZED_RELEASE} ${OPTIMIZED_DEBUG}
echo -e "${GREEN}✓ Cleaned${NC}"
echo ""

# Build all targets
echo -e "${BLUE}--- Building Naive GEMM ---${NC}"
build_target ${NAIVE_SRC} ${NAIVE_RELEASE} "${RELEASE_FLAGS}" "Release"
build_target ${NAIVE_SRC} ${NAIVE_DEBUG} "${DEBUG_FLAGS}" "Debug"

echo -e "${BLUE}--- Building Optimized GEMM ---${NC}"
build_target ${OPTIMIZED_SRC} ${OPTIMIZED_RELEASE} "${RELEASE_FLAGS}" "Release"
build_target ${OPTIMIZED_SRC} ${OPTIMIZED_DEBUG} "${DEBUG_FLAGS}" "Debug"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All builds completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# List built binaries with sizes
echo -e "${BLUE}Built binaries:${NC}"
ls -lh ${NAIVE_RELEASE} ${NAIVE_DEBUG} ${OPTIMIZED_RELEASE} ${OPTIMIZED_DEBUG} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Parse command line arguments for matrix dimensions
M=${1:-1024}
N=${2:-1536}
K=${3:-7168}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Running Release Binaries${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Matrix dimensions: M=${M}, N=${N}, K=${K}${NC}"
echo ""

# Run naive version
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Running Naive GEMM (Release)...${NC}"
echo -e "${YELLOW}========================================${NC}"
./${NAIVE_RELEASE} ${M} ${N} ${K}
echo ""

# Run optimized version
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Running Optimized GEMM (Release)...${NC}"
echo -e "${YELLOW}========================================${NC}"
./${OPTIMIZED_RELEASE} ${M} ${N} ${K}
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All tests completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}To run debug versions:${NC}"
echo -e "  ./${NAIVE_DEBUG} ${M} ${N} ${K}"
echo -e "  ./${OPTIMIZED_DEBUG} ${M} ${N} ${K}"
echo ""
echo -e "${BLUE}To run with custom dimensions:${NC}"
echo -e "  ./build_run.sh <M> <N> <K>"
echo -e "  Example: ./build_run.sh 2048 2048 2048"
