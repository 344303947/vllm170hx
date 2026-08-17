#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash-0731 @ CMP 170HX 一键启动（PP3/PP4 + DSpark, conda env vllm26）
#   ◆ 参考: github.com/allover326/deepseek-v4-cmp170hx/issues/2 (3x PP3) 与 170hx4 PP4
#   ◆ 环境: 独立 conda env vllm26（不与 170hx3/170hx4 共用）
# 用法: ./start_vllm_170hx.sh [选项]
#   --pp N / -p N  并行度: 3=自动选卡(3卡机用 0,1,2；4卡机用 0,1,2) (part 15,16,12)；4=卡 0,1,2,3 (part 12,12,12,7) maxlen 1M（默认 3）
#   --gpu-util N  覆盖 gpu_memory_utilization（默认 3卡=0.95 / 4卡=0.93）
#   --max-batched N 覆盖 max-num-batched-tokens（默认 2048；试 4096/8192 消除调度告警）
#   --max-seqs N    覆盖 max-num-seqs（默认 8；KV 允许时试 16 提聚合吞吐）
#   --marlin-atomic 设 VLLM_MARLIN_USE_ATOMIC_ADD=1（SM80 fp8 Marlin 小 size_n 提速）
#   --plain       去掉 DSpark 投机解码（decode 中文 ~58 tok/s 的可复现模式）
#   --maxlen N    上下文长度（默认统一 1048576；3卡显存紧张时可用 --maxlen 调低）
#   --port P      API 端口（默认 9003）
#   --name N      served-model-name（默认 DeepSeek-V4-Flash-0731）
#   --ab / --abliterated 使用固态盘上 abliterated 版本（默认模式，NVMe 加速加载 /models_ssd/...-abliterated）
#   --base        改用基础模型 /model/DeepSeek-V4-Flash-0731（默认是 abliterated，--base 切回）
#   --bg          后台运行（默认前台直接显示日志）
#   --force       启动前强制清理本 env 的旧实例（默认绝不杀任何进程）
#   --no-thinking 关闭推理思考（默认已开启；用此参数关闭后响应直接输出）
#   --think      开启推理思考（首版默认 enable_thinking=true 基础模式）
#   --effort E    思考强度: high=普通思考 / max=最大深度思考（tokenizer 注入更强思考指令，更慢更耗 token）/ none=关闭（默认 high；none 等同 --no-thinking）
# 日志:   ~/logs/vllm26_*.log
# 地址:   http://0.0.0.0:$PORT/v1  (api-key: 脚本内 API_KEY 变量)
# =============================================================================
set -euo pipefail

# ---- 目录/固定约定 -------------------------------------------------------------
ENV_NAME=vllm26
VLLM_ROOT=/home/sean/works/vllm26/vllm
MODEL_DIR=/model/DeepSeek-V4-Flash-0731
MODEL_ABLITERATED=/models_ssd/DeepSeek-V4-Flash-0731-abliterated   # 固态盘(NVMe)加速加载
ABLITERATED=1                        # 默认使用 abliterated 版本模型；--base 切换为基础版
ROW_CHUNK=64                          # 长对话安全的上下文天花板修复值
API_KEY='sk_344303'  #sk_344303

# ---- 可配置项 -----------------------------------------------------------------
PORT=9003
MAXLEN=1048576                    # 默认上下文 1M（3卡/4卡、ab/base 统一；可用 --maxlen 覆盖）
MAXLEN_SET=0                      # 用户是否显式给了 --maxlen
PP=3                             # 3 或 4
NAME=DeepSeek-V4-Flash-0731
SPEC='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
FG=1
THINKING=1                        # 默认开启推理思考；--no-thinking 关闭（响应直接输出）
EFFORT=high                       # 默认思考强度: high/max/none（--effort none 关思考）
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
    --no-thinking) THINKING=0; shift ;;
    --think|--thinking) THINKING=1; shift ;;
    --effort)   EFFORT="$2"; shift 2 ;;
    --maxlen)   MAXLEN="$2"; MAXLEN_SET=1; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --name)   NAME="$2"; shift 2 ;;
    --ab|--abliterated) ABLITERATED=1; shift ;;
    --base)     ABLITERATED=0; shift ;;
    --bg)     FG=0; shift ;;
    --force)  FORCE=1; shift ;;
    -h|--help|-help)
      sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
      exit 0;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# ---- 工具函数 --------------------------------------------------------------------
log()  { echo -e "$(date '+%F %T') [INFO]  $*"; }
die()  { echo -e "$(date '+%F %T') [ERROR] $*" >&2; exit 1; }

case "$EFFORT" in
  high|max|none) ;;
  *) die "未知 --effort=$EFFORT（仅支持 high/max/none）" ;;
esac
[ "$EFFORT" = "none" ] && THINKING=0    # --effort none 等同 --no-thinking（帮助注释语义）

# ---- 0. GPU 数量检测（PP3 按机器自动选卡）--------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "未找到 nvidia-smi"
fi
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1 || echo 0)
GPU_COUNT=${GPU_COUNT:-0}

# ---- 按 PP 解析卡集/分区/显存利用率/上下文长度 --------------------------------------
case "$PP" in
  3)
    if [ "$GPU_COUNT" -ge 4 ]; then
      # 4 卡机器: 默认用前面的 0,1,2
      GPU_IDS=(0 1 2)
      GPU_LIST="0,1,2"
    elif [ "$GPU_COUNT" -ge 3 ]; then
      # 3 卡机器: 用 0,1,2
      GPU_IDS=(0 1 2)
      GPU_LIST="0,1,2"
    else
      die "PP=3 需要至少 3 张 GPU，当前仅检测到 $GPU_COUNT 张"
    fi
    PARTITION="15,16,12"          # 43 层 / 3 rank 求和=43（issue#2 实测可用）
    GPU_UTIL=0.95                 # 3卡 KV 显存紧张；0.95 下默认上下文 1M（1048576）
    [ "$MAXLEN_SET" = "0" ] && MAXLEN=1048576    # 3卡默认上下文 1M（与 4卡/ab 统一）
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

# 切换模型版本（默认 abliterated；--base 使用基础模型）
if [ "$ABLITERATED" = "1" ]; then
  MODEL_DIR="$MODEL_ABLITERATED"          # 默认：固态盘 abliterated 版本（NVMe 加速加载）
  [ "$MAXLEN_SET" = "0" ] && MAXLEN=1048576   # abliterated 版本默认 1M（可用 --maxlen 覆盖）
  # served-model-name 默认仍为 $NAME，如需区分可加 --name DeepSeek-V4-Flash-0731-abliterated
else
  log "使用基础模型: $MODEL_DIR"
fi

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
log "启动 DeepSeek-V4-Flash (PP${PP}${SPEC:+ + DSpark})  GPU=${GPU_IDS[*]}  端口=$PORT  maxlen=$MAXLEN  model=$MODEL_DIR"
log "VLLM_PP_LAYER_PARTITION=$PARTITION  DSV4_LOGITS_ROW_CHUNK=$ROW_CHUNK  gpu-util=$GPU_UTIL"

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --served-model-name "$NAME"
  --api-key "$API_KEY"
  --host 0.0.0.0
  --port "$PORT"
  --pipeline-parallel-size "$PP"
  --kv-cache-dtype fp8_ds_mla
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
  --generation-config vllm
  --override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0}'
  --trust-remote-code
)
if [ -n "$SPEC" ]; then
  CMD+=(--speculative-config "$SPEC")
fi
if [ "$THINKING" = "1" ] && [ "$EFFORT" != "none" ]; then
  CMD+=(--default-chat-template-kwargs "{\"enable_thinking\": true, \"reasoning_effort\": \"$EFFORT\"}")
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
