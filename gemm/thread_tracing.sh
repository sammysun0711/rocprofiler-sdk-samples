#!/bin/bash

# Thread tracing script for FP8 GEMM implementations
# Uses rocprofv3 with ATT (Async Trace Tool) for detailed GPU kernel analysis

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Binaries to trace
NAIVE_BIN="naive_gemm_debug"
OPTIMIZED_BIN="optimized_gemm_debug"

# Results directories
NAIVE_RESULTS="naive_gemm_trace"
OPTIMIZED_RESULTS="optimized_gemm_trace"

# rocprofv3 configuration
ROCPROF_CMD="rocprofv3"
ROCPROF_FLAGS="--att --att-activity 10"

# Matrix dimensions (can be overridden via command line)
M=${1:-1024}
N=${2:-1536}
K=${3:-7168}

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  FP8 GEMM Thread Tracing${NC}"
echo -e "${CYAN}  Using rocprofv3 with ATT${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${BLUE}Matrix dimensions: M=${M}, N=${N}, K=${K}${NC}"
echo -e "${BLUE}Profiler: ${ROCPROF_CMD}${NC}"
echo -e "${BLUE}Flags: ${ROCPROF_FLAGS}${NC}"
echo ""

# Check if rocprofv3 is available
if ! command -v ${ROCPROF_CMD} &> /dev/null; then
    echo -e "${RED}Error: ${ROCPROF_CMD} not found!${NC}"
    echo -e "${YELLOW}Please ensure ROCm profiling tools are installed.${NC}"
    echo -e "${YELLOW}Try: export PATH=/opt/rocm/bin:\$PATH${NC}"
    exit 1
fi

# Check if binaries exist
if [ ! -f "${NAIVE_BIN}" ]; then
    echo -e "${RED}Error: ${NAIVE_BIN} not found!${NC}"
    echo -e "${YELLOW}Please run ./build_run.sh first to build the binaries.${NC}"
    exit 1
fi

if [ ! -f "${OPTIMIZED_BIN}" ]; then
    echo -e "${RED}Error: ${OPTIMIZED_BIN} not found!${NC}"
    echo -e "${YELLOW}Please run ./build_run.sh first to build the binaries.${NC}"
    exit 1
fi

# Function to run profiling
run_profile() {
    local binary=$1
    local results_dir=$2
    local description=$3
    
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}  Tracing: ${description}${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${YELLOW}Binary: ${binary}${NC}"
    echo -e "${YELLOW}Results directory: ${results_dir}${NC}"
    echo ""
    
    # Remove old results
    if [ -d "${results_dir}" ]; then
        echo -e "${YELLOW}Removing old trace results...${NC}"
        rm -rf "${results_dir}"
    fi
    
    # Run profiling
    echo -e "${CYAN}Running profiler...${NC}"
    echo -e "${CYAN}Command: ${ROCPROF_CMD} ${ROCPROF_FLAGS} -d ./${results_dir} -- ./${binary} ${M} ${N} ${K}${NC}"
    echo ""
    
    if ${ROCPROF_CMD} ${ROCPROF_FLAGS} -d ./${results_dir} -- ./${binary} ${M} ${N} ${K}; then
        echo ""
        echo -e "${GREEN}✓ Profiling completed successfully!${NC}"
        echo -e "${GREEN}Results saved to: ${results_dir}/${NC}"
        
        # Show directory contents
        if [ -d "${results_dir}" ]; then
            echo -e "${BLUE}Generated files:${NC}"
            find "${results_dir}" -type f | head -20 | while read file; do
                size=$(du -h "$file" | cut -f1)
                echo -e "  ${file} (${size})"
            done
            
            total_files=$(find "${results_dir}" -type f | wc -l)
            if [ ${total_files} -gt 20 ]; then
                echo -e "  ... and $((total_files - 20)) more files"
            fi
            
            # Show total size
            total_size=$(du -sh "${results_dir}" | cut -f1)
            echo -e "${BLUE}Total size: ${total_size}${NC}"
        fi
    else
        echo ""
        echo -e "${RED}✗ Profiling failed!${NC}"
        return 1
    fi
    
    echo ""
}

# Trace Naive GEMM
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 1/2: Tracing Naive GEMM${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
run_profile "${NAIVE_BIN}" "${NAIVE_RESULTS}" "Naive FP8 GEMM (Unoptimized)"

# Trace Optimized GEMM
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 2/2: Tracing Optimized GEMM${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
run_profile "${OPTIMIZED_BIN}" "${OPTIMIZED_RESULTS}" "Optimized FP8 GEMM (MFMA)"

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Thread Tracing Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Results Summary:${NC}"
echo ""

if [ -d "${NAIVE_RESULTS}" ]; then
    naive_size=$(du -sh "${NAIVE_RESULTS}" | cut -f1)
    naive_files=$(find "${NAIVE_RESULTS}" -type f | wc -l)
    echo -e "${YELLOW}Naive GEMM:${NC}"
    echo -e "  Directory: ${NAIVE_RESULTS}"
    echo -e "  Files: ${naive_files}"
    echo -e "  Size: ${naive_size}"
    echo ""
fi

if [ -d "${OPTIMIZED_RESULTS}" ]; then
    opt_size=$(du -sh "${OPTIMIZED_RESULTS}" | cut -f1)
    opt_files=$(find "${OPTIMIZED_RESULTS}" -type f | wc -l)
    echo -e "${YELLOW}Optimized GEMM:${NC}"
    echo -e "  Directory: ${OPTIMIZED_RESULTS}"
    echo -e "  Files: ${opt_files}"
    echo -e "  Size: ${opt_size}"
    echo ""
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Analyzing Trace Results${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Look for key trace files
echo -e "${BLUE}Key trace files:${NC}"
echo ""

for results_dir in "${NAIVE_RESULTS}" "${OPTIMIZED_RESULTS}"; do
    if [ -d "${results_dir}" ]; then
        echo -e "${YELLOW}${results_dir}/${NC}"
        
        # Find ATT files
        att_files=$(find "${results_dir}" -name "*.att" 2>/dev/null | wc -l)
        if [ ${att_files} -gt 0 ]; then
            echo -e "  ATT trace files: ${att_files}"
            find "${results_dir}" -name "*.att" 2>/dev/null | head -5 | sed 's/^/    /'
            if [ ${att_files} -gt 5 ]; then
                echo "    ... and $((att_files - 5)) more"
            fi
        fi
        
        # Find results database
        if [ -f "${results_dir}"/*/*.db ]; then
            echo -e "  Results database: ✓"
        fi
        
        # Find JSON files
        json_files=$(find "${results_dir}" -name "*.json" 2>/dev/null | wc -l)
        if [ ${json_files} -gt 0 ]; then
            echo -e "  JSON metadata files: ${json_files}"
        fi
        
        # Find CSV files
        csv_files=$(find "${results_dir}" -name "*.csv" 2>/dev/null | wc -l)
        if [ ${csv_files} -gt 0 ]; then
            echo -e "  CSV stats files: ${csv_files}"
            find "${results_dir}" -name "*.csv" 2>/dev/null | sed 's/^/    /'
        fi
        
        echo ""
    fi
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Analysis Instructions${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${BLUE}To analyze the trace results:${NC}"
echo ""
echo -e "1. View with Omniperf (if available):"
echo -e "   ${YELLOW}omniperf analyze -p ${NAIVE_RESULTS}${NC}"
echo -e "   ${YELLOW}omniperf analyze -p ${OPTIMIZED_RESULTS}${NC}"
echo ""
echo -e "2. View CSV statistics:"
echo -e "   ${YELLOW}cat ${NAIVE_RESULTS}/*.csv${NC}"
echo -e "   ${YELLOW}cat ${OPTIMIZED_RESULTS}/*.csv${NC}"
echo ""
echo -e "3. View JSON metadata:"
echo -e "   ${YELLOW}ls ${NAIVE_RESULTS}/*/code.json${NC}"
echo -e "   ${YELLOW}ls ${OPTIMIZED_RESULTS}/*/code.json${NC}"
echo ""
echo -e "4. Compare kernel execution:"
echo -e "   ${YELLOW}diff -r ${NAIVE_RESULTS} ${OPTIMIZED_RESULTS}${NC}"
echo ""
echo -e "5. Extract ATT files for detailed wavefront analysis:"
echo -e "   ${YELLOW}find ${OPTIMIZED_RESULTS} -name '*.att'${NC}"
echo ""
echo -e "${GREEN}Thread tracing completed successfully!${NC}"
