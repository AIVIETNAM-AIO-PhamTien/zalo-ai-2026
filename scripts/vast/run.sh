#!/usr/bin/env bash
# Bật vLLM (nếu chưa chạy), chờ sẵn sàng, rồi chạy agent trên một split.
#   bash scripts/vast/run.sh public                     # data/public_test → outputs/public/submission.jsonl
#   bash scripts/vast/run.sh train 200                  # 200 câu train mẫu → chấm PA
#   MODEL_PATH=models/Qwen3.5-9B bash scripts/vast/run.sh public
# Tham số thêm cho datamate.run đặt trong EXTRA, vd EXTRA="--samples 3 --no-thinking"
set -euo pipefail
cd "$(dirname "$0")/../.."

SPLIT=${1:-public}
N=${2:-200}
MODEL_PATH=${MODEL_PATH:-models/Qwen3-14B}
NAME=${SERVED_NAME:-$(basename "$MODEL_PATH")}
PORT=${PORT:-8000}
EXTRA=${EXTRA:-}
TAG=${TAG:-$NAME}
SEED=${SEED:-0}  # bộ câu train lấy mẫu; đổi seed để có bộ câu khác

if ! curl -sf "http://localhost:$PORT/health" >/dev/null; then
  mkdir -p outputs
  echo "[run.sh] khởi động vLLM… (log: outputs/vllm.log)"
  MODEL_PATH="$MODEL_PATH" SERVED_NAME="$NAME" PORT="$PORT" nohup bash scripts/vast/serve.sh > outputs/vllm.log 2>&1 &
  for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null && break
    sleep 5
  done
  curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "vLLM không lên, xem outputs/vllm.log"; exit 1; }
fi

COMMON=(--llm-url "http://localhost:$PORT/v1" --model "$NAME" --cache outputs/cache)
case "$SPLIT" in
  public)
    OUT=outputs/public_$TAG
    python3 -m datamate.run --data data/public_test --out "$OUT/submission.jsonl" --logs "$OUT/logs" "${COMMON[@]}" $EXTRA
    python3 scripts/check_submission.py "$OUT/submission.jsonl" data/public_test/questions.jsonl
    echo "Nộp file: $OUT/submission.jsonl"
    ;;
  train)
    OUT=outputs/train_${TAG}_n$N
    mkdir -p "$OUT"
    if [ -n "${IDS:-}" ]; then cp "$IDS" "$OUT/ids.txt"; else python3 scripts/sample_ids.py data/train "$N" "$SEED" > "$OUT/ids.txt"; fi
    python3 -m datamate.run --data data/train --out "$OUT/submission.jsonl" --logs "$OUT/logs" --ids "$OUT/ids.txt" "${COMMON[@]}" $EXTRA
    python3 scripts/score.py "$OUT/submission.jsonl" data/train/answers.jsonl --ids "$OUT/ids.txt"
    ;;
  *) echo "split phải là public hoặc train"; exit 2 ;;
esac
