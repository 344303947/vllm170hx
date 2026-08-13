#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DeepSeek-V4-Flash-0731 @ 4x CMP 170HX (GA100 sm_80, 64GB) — conda 裸机启动脚本
# 基于 allover326/deepseek-v4-cmp170hx 实战定稿版参数
# 要求：4 张 GPU 全部在线（nvidia-smi 确认），不足 4 卡时不要启动
# =============================================================================

cd "$(dirname "${BASH_SOURCE[0]}")/.."    # 切到 vLLM 仓库根，避免 CWD 干扰 import vllm

export HF_HUB_OFFLINE=1
export VLLM_PP_LAYER_PARTITION=12,12,12,7
export DSV4_LOGITS_ROW_CHUNK=128          # one-shot 满 1M；对话场景改 64
export CUDA_HOME=/usr/local/cuda

MODEL_DIR=/models/models/DeepSeek-V4-Flash-0731
PORT=9070
SERVED_NAME=/DeepSeek-V4-Flash-0731
API_KEY=sk_344303

NVGPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NVGPU_COUNT" -lt 4 ]; then
    echo "FATAL: 检测到 $NVGPU_COUNT 张 GPU，DeepSeek-V4-Flash 需要 4 张（PP4）。"
    echo "       恢复命令: nvidia-smi -r -i 0,1,2,3 或 rmmod nvidia_uvm && modprobe nvidia_uvm"
    exit 1
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4

# =============================================================================
# 可选 DSpark 投机解码（目标模型权重 /models/models/DeepSeek-V4-Flash-DSpark 时）
#   加 --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
#   注意: num_speculative_tokens 必须严格 = 5；temp 0 下输出不可复现
# 红线（严禁添加）:
#   --enforce-eager / --attention-backend flashinfer / --quantization
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments
# =============================================================================
