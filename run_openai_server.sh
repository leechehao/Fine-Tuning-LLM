python -m vllm.entrypoints.openai.api_server \
    --model model_16bit \
    --host 0.0.0.0 \
    --port 7788 \
    --gpu-memory-utilization 0.65 \
    --max-model-len 2000
    