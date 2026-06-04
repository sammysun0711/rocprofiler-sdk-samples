export ROCPROFILER_REGISTER_LOG_LEVEL=info
export ROCP_TOOL_ATTACH=1
export HIP_VISIBLE_DEVICES=6,7
export VLLM_ROCM_USE_AITER=1

vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization=0.7 \
    --port=20010 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 14096 \
    --uvicorn-log-level=error 2>&1 | tee vllm_server-info.log
