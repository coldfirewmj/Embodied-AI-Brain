conda activate qwen3_vllm
# 设置环境变量，指定使用两张卡
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASHINFER

vllm serve "/data2/modelscope_models/Qwen3-VL-8B-Instruct" \
    --tensor-parallel-size 2 \
    --served-model-name Qwen3-VL-8B-Instruct \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 10, "video": 1}' \
    --enforce-eager