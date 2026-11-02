#!/usr/bin/env bash
# Khởi động vLLM (API tương thích OpenAI) ở cổng 8000, model chia đôi trên 2 GPU (tensor parallel).
#   bash scripts/vast/serve.sh                          # models/Qwen3-14B
#   MODEL_PATH=models/Qwen3.5-9B bash scripts/vast/serve.sh
#   MODEL_PATH=models/Qwen3-14B-FP8 DP=2 TP=1 bash scripts/vast/serve.sh   # mỗi GPU một bản FP8
# Qwen3-14B bf16 ~27.6 GiB: một bản trên một card 32 GB (DP=2) không còn chỗ cho KV cache, vLLM không lên.
# TP=2 → mỗi card ~13.8 GiB weights, ~13 GiB KV (~170k token tổng).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_PATH=${MODEL_PATH:-models/Qwen3-14B}
NAME=${SERVED_NAME:-$(basename "$MODEL_PATH")}
DP=${DP:-1}
TP=${TP:-2}
PORT=${PORT:-8000}
# Prompt đầu (rulebook + README + brief) ~7–13k token, cộng các lượt code và thinking.
# Qwen3 config cho phép 40960 token; vLLM 0.32 không nhận override rope_scaling cũ nên YaRN mặc định tắt
# (YaRN tĩnh cũng làm giảm chất lượng ở prompt ngắn). Qwen3.5 gốc 256k.
EXTRA_ARGS=()
case "${YARN:-0}:$NAME" in
  1:Qwen3-*)
    MAXLEN=${MAXLEN:-65536}
    EXTRA_ARGS+=(--hf-overrides '{"rope_scaling": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}}')
    ;;
  *:Qwen3-*) MAXLEN=${MAXLEN:-40960} ;;
  # model khác (vd DataMind-14B nền Qwen2.5: 32768): lấy giới hạn trong config.json, đặt cao hơn thì vLLM không lên
  *) MAXLEN=${MAXLEN:-$(python3 -c "import json,sys; c=json.load(open(sys.argv[1]+'/config.json')); print(min(65536, c.get('max_position_embeddings', 32768)))" "$MODEL_PATH" 2>/dev/null || echo 32768)} ;;
esac

# máy không có GPU P2P (vd 2×5090 cùng NUMA node, PCIe): vLLM treo lúc dựng CUDA graph → chạy với
#   NCCL_P2P_DISABLE=1 VLLM_EXTRA="--disable-custom-all-reduce" (thêm --enforce-eager nếu vẫn treo)
if [ "$TP" -gt 1 ] && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GeForce \
   && [ -z "${VLLM_EXTRA:-}" ]; then
  export NCCL_P2P_DISABLE=1  # card GeForce không có P2P (driver khoá); bật thử thì treo (lượt GPU 2, 2×5090)
  VLLM_EXTRA="--disable-custom-all-reduce"
fi
read -ra MORE <<< "${VLLM_EXTRA:-}"
EXTRA_ARGS+=("${MORE[@]}")

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  "${EXTRA_ARGS[@]}" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --seed 0 \
  --port "$PORT"
