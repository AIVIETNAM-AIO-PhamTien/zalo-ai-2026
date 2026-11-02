#!/usr/bin/env bash
# Điểm vào khi chấm: đọc test ở /data, ghi /result/submission.jsonl và log từng câu ở /result/logs.
# Mạng tắt, /data read-only: cache và file tạm ghi vào /tmp.
set -uo pipefail
cd "$(dirname "$0")"

MODEL_PATH=${MODEL_PATH:-/models/Qwen3-14B}
NAME=$(basename "$MODEL_PATH")
# cấu hình agent chốt sau lượt GPU kiểm chứng; đổi bằng biến môi trường khi build / chạy
AGENT_FLAGS=${AGENT_FLAGS:---shots datamate/shots.jsonl --reconcile --diverse --plan --cards auto --sem-facts}
mkdir -p /result/logs
START=$(date +%s)
nvidia-smi > /result/logs/nvidia-smi.txt 2>&1

MODEL_PATH="$MODEL_PATH" SERVED_NAME="$NAME" bash scripts/vast/serve.sh > /result/logs/vllm.log 2>&1 &
VLLM_PID=$!
READY=0
for i in $(seq 1 240); do
  if curl -sf http://localhost:8000/health >/dev/null; then READY=1; break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || break  # vLLM chết khi khởi động: không chờ hết 20 phút
  sleep 5
done
# vLLM không lên vẫn chạy tiếp: tầng tất định trả lời được một phần, câu còn lại thành abstain
[ "$READY" = 1 ] || echo "[predict] vLLM không lên sau $(( $(date +%s) - START ))s, xem vllm.log" | tee -a /result/logs/run.log

# ngân sách 6 giờ tính cả lúc nạp model: chừa 20 phút an toàn
USED=$(( $(date +%s) - START ))
BUDGET=$(( 6 * 3600 - 1200 - USED ))
echo "[predict] khởi động mất ${USED}s (vLLM ready=$READY), ngân sách còn ${BUDGET}s" | tee -a /result/logs/run.log

python3 -m datamate.run --data /data --out /result/submission.jsonl --logs /result/logs \
  --cache /tmp/datamate_cache --llm-url http://localhost:8000/v1 --model "$NAME" \
  --time-budget "$BUDGET" $AGENT_FLAGS 2>&1 | tee -a /result/logs/run.log
