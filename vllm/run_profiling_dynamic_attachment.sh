export ROCPROFILER_REGISTER_LOG_LEVEL=info
export ROCP_TOOL_ATTACH=1

PID=$(pgrep -f "VLLM::Worker_TP" | head -1)
if [ -z "$PID" ]; then
    echo "Error: no VLLM::Worker_TP process found" >&2
    exit 1
fi
echo "Attaching to VLLM::Worker_TP PID=$PID"

rocprofv3 --attach "$PID"  \
    --hip-trace \
    --sys-trace \
    --attach-duration-msec 5000 \
    --output-format pftrace \
    -o out 2>&1 | tee rocprofv3.log
