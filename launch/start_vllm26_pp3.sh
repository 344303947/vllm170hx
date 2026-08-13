#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash-0731 @ CMP 170HX 一键启动（PP3/PP4 + DSpark, conda env vllm26）
#   ◆ 参考: github.com/allover326/deepseek-v4-cmp170hx/issues/2 (3x PP3) 与 170hx4 PP4
#   ◆ 环境: 独立 conda env vllm26（不与 170hx3/170hx4 共用）
# 用法: ./start_vllm26_pp3.sh [选项]
#   --pp N / -p N  并行度: 3=自动选卡(3卡机用 0,1,2；4卡机用 1,2,3 保留 0) (part 16,16,11)；4=卡 0,1,2,3 (part 12,12,12,7) maxlen 1M（默认 3）
#   --gpu-util N  覆盖 gpu_memory_utilization（默认 3卡=0.95 / 4卡=0.93）
#   --max-batched N 覆盖 max-num-batched-tokens（默认 2048；试 4096/8192 消除调度告警）
#   --max-seqs N    覆盖 max-num-seqs（默认 8；KV 允许时试 16 提聚合吞吐）
#   --marlin-atomic 设 VLLM_MARLIN_USE_ATOMIC_ADD=1（SM80 fp8 Marlin 小 size_n 提速）
#   --plain       去掉 DSpark 投机解码（decode 中文 ~58 tok/s 的可复现模式）
#   --maxlen N    上下文长度（覆盖按 PP 的默认值：3卡=700000 / 4卡=1048576）
#   --port P      API 端口（默认 9070）
#   --name N      served-model-name（默认 DeepSeek-V4-Flash-0731）
#   --bg          后台运行（默认前台直接显示日志）
#   --force       启动前强制清理本 env 的旧实例（默认绝不杀任何进程）
# 日志:   ~/logs/vllm26_*.log
# 地址:   http://0.0.0.0:$PORT/v1  (api-key: sk_vllm26)
# =============================================================================
set -euo pipefail

# ---- 目录/固定约定 -------------------------------------------------------------
ENV_NAME=vllm26
# 由脚本所在位置推导 vLLM 仓库根（本脚本位于 <repo>/launch/ 下），移动整个仓库目录也能正常启动
VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR=/model/DeepSeek-V4-Flash-0731
ROW_CHUNK=64                          # 长对话安全的上下文天花板修复值
REPO_SERVED_PREFIX=1                   # served-model-name 是否带 "/" 前缀（DeepSeek-V4 专用）
API_KEY='sk-gRSilwwHpck1glDE9a40A435EcB04353957444F4Ad836807'  #sk_344303

# ---- 可配置项 -----------------------------------------------------------------
PORT=9003 #9070
MAXLEN=1048576                    # 占位；未显式指定时由下方按 PP 覆盖（3卡=970K / 4卡=1M）
MAXLEN_SET=0                      # 用户是否显式给了 --maxlen
PP=3                             # 3 或 4
NAME=DeepSeek-V4-Flash-0731
SPEC='{"method":"dspark","num_speculative_tokens":5}'
FG=1
FORCE=0
GPU_UTIL_OVERRIDE=""
MAX_BATCHED=""                # 覆盖 --max-num-batched-tokens（默认 2048；消除 max_num_scheduled_tokens 告警可试 4096/8192）
MAX_SEQS=""                  # 覆盖 --max-num-seqs（默认 8；KV 允许时提并发可试 16）
MARLIN_ATOMIC=0            # 1=export VLLM_MARLIN_USE_ATOMIC_ADD（SM80 fp8 Marlin 小 size_n 提速提示）
LOG_DIR="$HOME/logs"
LOG_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--pp)    PP="$2"; shift 2 ;;
    --gpu-util) GPU_UTIL_OVERRIDE="$2"; shift 2 ;;
    --max-batched) MAX_BATCHED="$2"; shift 2 ;;
    --max-seqs)   MAX_SEQS="$2"; shift 2 ;;
    --marlin-atomic) MARLIN_ATOMIC=1; shift ;;
    --plain)    SPEC=""; shift ;;
    --maxlen)   MAXLEN="$2"; MAXLEN_SET=1; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --name)   NAME="$2"; shift 2 ;;
    --bg)     FG=0; shift ;;
    --force)  FORCE=1; shift ;;
    -h|--help|-help)
      sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# ---- 工具函数 --------------------------------------------------------------------
log()  { echo -e "$(date '+%F %T') [INFO]  $*"; }
die()  { echo -e "$(date '+%F %T') [ERROR] $*" >&2; exit 1; }

# ---- 0. GPU 数量检测（PP3 按机器自动选卡）--------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "未找到 nvidia-smi"
fi
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1)
GPU_COUNT=${GPU_COUNT:-0}

# ---- 按 PP 解析卡集/分区/显存利用率/上下文长度 --------------------------------------
case "$PP" in
  3)
    if [ "$GPU_COUNT" -ge 4 ]; then
      # 4 卡机器: 用 1,2,3，保留 0 号卡（0 常被其它实例占用）
      GPU_IDS=(1 2 3)
      GPU_LIST="1,2,3"
    elif [ "$GPU_COUNT" -ge 3 ]; then
      # 3 卡机器: 用 0,1,2
      GPU_IDS=(0 1 2)
      GPU_LIST="0,1,2"
    else
      die "PP=3 需要至少 3 张 GPU，当前仅检测到 $GPU_COUNT 张"
    fi
    PARTITION="16,16,11"          # 43 层 / 3 rank 求和=43（issue#2 实测可用）
    GPU_UTIL=0.95                 # 3卡 KV 显存紧张；0.95 下默认上下文 970K
    [ "$MAXLEN_SET" = "0" ] && MAXLEN=700000    # 3卡默认上下文 700000K 489216
    ;;
  4)
    GPU_IDS=(0 1 2 3)
    GPU_LIST="0,1,2,3"
    PARTITION="12,12,12,7"        # 43 层 / 4 rank 求和=43（PP4+DSpark 参考值）
    GPU_UTIL=0.93
    [ "$MAXLEN_SET" = "0" ] && MAXLEN=1048576   # 4卡默认 1M
    ;;
  *)
    echo "未知 --pp=$PP（仅支持 3 或 4）" >&2; exit 1
    ;;
esac

# 用户可用 --gpu-util 覆盖默认利用率（如 3 卡 1M 仍不够时调高，或 4 卡调低保稳定）
if [ -n "$GPU_UTIL_OVERRIDE" ]; then
  GPU_UTIL="$GPU_UTIL_OVERRIDE"
fi

# ---- 0. 日志 ---------------------------------------------------------------------
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm26_$(date +%Y%m%d_%H%M%S).log"

# ---- 1. 预检 ---------------------------------------------------------------------
# 1a. 目标卡显存占用检测（默认只警告并退出，绝不杀进程）
BUSY=""
for g in "${GPU_IDS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
  if [ -n "$used" ] && [ "$used" -gt 4096 ]; then
    BUSY="$BUSY $g(${used}MiB)"
  fi
done
if [ -n "$BUSY" ]; then
  echo "——————————————————————————————————————————————————————————————————————"
  echo "  GPU ${GPU_IDS[*]} 当前显存占用，可能被其它实例占用:${BUSY}"
  echo "  vllm26 启动会在这些卡上 OOM，默认【不代劳终止任何进程】。"
  echo "  请先手动停止占用卡片的旧实例后重试；"
  echo "  确认无误可加 --force（仅清理本 env vllm26 的旧 vllm 实例）。"
  echo "——————————————————————————————————————————————————————————————————————"
  if [ "$FORCE" = "1" ]; then
    log "已确认 --force，继续（先清理本 env 旧实例）..."
  else
    exit 1
  fi
fi

# ---- 2. 环境 ----------------------------------------------------------------------
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
cd "$VLLM_ROOT"     # ★ 必须 cd 到 repo 根，避免父目录 vllm/ 命名空间包遮蔽

if ! python -c "import vllm" >/dev/null 2>&1; then
  die "env $ENV_NAME 中 import vllm 失败，请先完成环境安装"
fi

# ---- 3. 环境变量 --------------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export VLLM_PP_LAYER_PARTITION="$PARTITION"
export DSV4_LOGITS_ROW_CHUNK="$ROW_CHUNK"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
if [ "$MARLIN_ATOMIC" = "1" ]; then
  export VLLM_MARLIN_USE_ATOMIC_ADD=1
fi
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# ---- 4. 清理本 env 的旧实例（仅 --force 且确实需要）------------------------------
if [ "$FORCE" = "1" ]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  sleep 2
fi

# ---- 5. 组装启动命令 -------------------------------------------------------------------
log "启动 DeepSeek-V4-Flash (PP${PP}${SPEC:+ + DSpark})  GPU=${GPU_IDS[*]}  端口=$PORT  maxlen=$MAXLEN"
log "VLLM_PP_LAYER_PARTITION=$PARTITION  DSV4_LOGITS_ROW_CHUNK=$ROW_CHUNK  gpu-util=$GPU_UTIL"

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --served-model-name "$NAME"
  --api-key "$API_KEY"
  --host 0.0.0.0
  --port "$PORT"
  --pipeline-parallel-size "$PP"
  --kv-cache-dtype fp8
  --block-size 256
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "${MAX_BATCHED:-2048}"
  --max-num-seqs "${MAX_SEQS:-8}"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching
  --no-enable-flashinfer-autotune
  --tokenizer-mode deepseek_v4
  --enable-auto-tool-choice
  --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4
  --trust-remote-code
)
if [ -n "$SPEC" ]; then
  CMD+=(--speculative-config "$SPEC")
fi

# ---- 6. 运行 ----------------------------------------------------------------------------
if [ "$FG" = "1" ]; then
  exec "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  VLLM_PID=$!
  log "PID=$VLLM_PID  日志: $LOG_FILE"
  log "等待就绪... "
  for i in $(seq 1 180); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vllm 进程提前退出，日志尾部：" >&2
      tail -n 30 "$LOG_FILE" >&2
      exit 1
    fi
    if curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "READY  http://127.0.0.1:$PORT/v1  (api-key: $API_KEY)"
      echo "测试: curl http://127.0.0.1:$PORT/v1/chat/completions -H \"Authorization: Bearer $API_KEY\" -H 'Content-Type: application/json' -d '{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
      exit 0
    fi
    if grep -q "Traceback\|FATAL\|WorkerProc failed\|CUDA error" "$LOG_FILE" 2>/dev/null; then
      echo "启动出错，日志尾部：" >&2
      tail -n 30 "$LOG_FILE" >&2
      exit 1
    fi
    sleep 5
  done
  echo "超时未就绪，请查看日志: $LOG_FILE" >&2
  exit 1
fi
