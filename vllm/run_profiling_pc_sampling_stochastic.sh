export ROCPROFILER_REGISTER_LOG_LEVEL=info
export ROCP_TOOL_ATTACH=1

PID=$(pgrep -f "VLLM::Worker_TP" | head -1)
if [ -z "$PID" ]; then
    echo "Error: no VLLM::Worker_TP process found" >&2
    exit 1
fi
echo "Attaching to VLLM::Worker_TP PID=$PID"

rocprofv3 --attach $PID                         \
          --pc-sampling-beta-enabled            \
          --pc-sampling-method  stochastic      \
          --pc-sampling-unit cycles             \
          --pc-sampling-interval 1048576        \
          --output-format csv                   \
          -d rocprofile_pc_sampling_stochastic  \
          2>&1 | tee pc_sampling_stochastic.log
