# 前期工作：
# 首先下载docker镜像：
# docker pull ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
# 其次建立docker：
# docker run --runtime nvidia -it  \
#     --name qwen3-vlm \
#     --pid=host \
#     --network host \
#     --shm-size=32gb \
#     --group-add video \
#     -v ~/data/models:/models \
#     ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
docker start qwen3-vlm
docker exec -it qwen3-vlm /bin/bash
vllm serve "/models/Qwen3-VL-8B-Instruct-4bit-GPTQ" \
    --served-model-name Qwen3-VL-8B-Instruct \
    --quantization gptq \
    --trust-remote-code \
    --dtype float16 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 1}' \
    --enforce-eager \
    --override-generation-config '{"max_window_size": 1920}'