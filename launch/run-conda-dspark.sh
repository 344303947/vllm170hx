#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash-0731 @ 4x CMP 170HX (GA100 sm_80, 64GB) — conda 裸机 + DSpark
# 对应 Docker 版 run-pp-dspark.sh 的最佳配置（PP4 + DSpark 投机解码）
#   decode 98.1 tok/s aggregate | prefill ~5,300 t/s | large KV pool
# 要求：4 张 GPU 全部在线（nvidia-smi 确认），不足 4 卡时不要启动
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."    # 切到 vLLM 仓库根，避免 CWD 干扰 import vllm

source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate dsv4

export HF_HUB_OFFLINE=1
export VLLM_PP_LAYER_PARTITION=12,12,12,7
export DSV4_LOGITS_ROW_CHUNK=64          # 对话与 one-shot 均安全；满 1M
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_DIR=/models/models/DeepSeek-V4-Flash-0731
PORT=9070
SERVED_NAME=/DeepSeek-V4-Flash-0731
API_KEY=sk_344303

NVGPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NVGPU_COUNT" -lt 4 ]; then
    echo "FATAL: 检测到 $NVGPU_COUNT 张 GPU，DeepSeek-V4-Flash 需要 4 张（PP4）。"
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
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}'