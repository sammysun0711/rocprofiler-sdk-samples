vllm bench serve  \
     --backend vllm  \
     --model /models/Qwen3-0.6B  \
     --trust-remote-code \
     --endpoint /v1/completions  \
     --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
     --num-prompts 1000  \
     --port 20010 \
     --max-concurrency 10 2>&1 | tee vllm_bench.log

#dmesg 2>&1 | tee dmesg.log
