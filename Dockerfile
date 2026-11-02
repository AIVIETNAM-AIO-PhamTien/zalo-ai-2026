# Image nộp bài: chạy offline hoàn toàn, đọc test ở /data (read-only), ghi /result/submission.jsonl + /result/logs.
#   docker build -t datamate --build-arg BASE=vllm/vllm-openai:<tag> .     # cần models/Qwen3-14B trong thư mục build
#   docker run --gpus all --network none -v <test>:/data:ro -v <out>:/result datamate
# BASE: theo Docker guideline của BTC (chưa công bố); tạm dùng image vLLM có PyTorch cho Blackwell (CUDA >= 12.8).
# Ghim tag cụ thể (không dùng latest) theo bản vLLM đã chạy được trên 2x RTX 5090.
# lượt chạy thật 27–28/10 dùng vLLM 0.32 (torch cu130) trên 2×5090; kiểm tag tồn tại trên Docker Hub khi build
ARG BASE=vllm/vllm-openai:v0.32.0
FROM ${BASE}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY datamate ./datamate
COPY scripts ./scripts
COPY predict.sh ./
RUN pip install --no-cache-dir . && chmod +x predict.sh

# weights để cuối: đổi code không phải build lại lớp model ~30 GB
COPY models/Qwen3-14B /models/Qwen3-14B

# mạng bị tắt khi chấm: không cho thư viện nào thử gọi ra ngoài
ENV MODEL_PATH=/models/Qwen3-14B \
    AGENT_FLAGS="--shots datamate/shots.jsonl --reconcile --diverse --plan --cards auto --sem-facts" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["bash", "/app/predict.sh"]
