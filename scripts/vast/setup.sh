#!/usr/bin/env bash
# Cài môi trường trên máy vast.ai (Ubuntu, driver hỗ trợ CUDA >= 12.8, 2x RTX 5090).
# Image gợi ý khi thuê: vllm/vllm-openai (đã có vLLM + PyTorch cho Blackwell) hoặc pytorch/pytorch cuda12.8.
#   bash scripts/vast/setup.sh                    # mặc định Qwen/Qwen3-14B
#   MODEL=Qwen/Qwen3.5-9B bash scripts/vast/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-Qwen/Qwen3-14B}
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate  # template PyTorch của vast: venv ở /venv/main
# PyPI có lúc chỉ ~0.3 MB/s trên máy vast; uv tải song song và lấy torch + thư viện CUDA từ kho PyTorch (~7 MB/s)
if command -v uv >/dev/null; then
  PIP=(uv pip install --python "$(command -v python3)" --extra-index-url https://download.pytorch.org/whl/cu130
       --index-strategy unsafe-best-match)
else
  python3 -m pip install -U pip
  PIP=(python3 -m pip install)
fi
"${PIP[@]}" -e ".[dev]"
python3 -c "import vllm" 2>/dev/null || "${PIP[@]}" -U vllm
"${PIP[@]}" -U huggingface_hub hf_transfer hf_xet || "${PIP[@]}" -U huggingface_hub

# tải nhanh: Xet chế độ hiệu năng cao (repo Xet) hoặc hf_transfer (repo LFS cũ), 16 file song song
export HF_XET_HIGH_PERFORMANCE=1
python3 -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1
t0=$(date +%s)
python3 scripts/vast/fetch_model.py "$MODEL" "models/$(basename "$MODEL")"
echo "tải $MODEL mất $(( $(date +%s) - t0 )) giây"
# tải sẵn model so sánh ở nền trong lúc chạy chặng ab (chặng model khỏi chờ); PREFETCH= để tắt
PREFETCH=${PREFETCH-zjunlp/DataMind-14B}
if [ -n "$PREFETCH" ] && [ ! -f "models/$(basename "$PREFETCH")/config.json" ]; then
  mkdir -p outputs
  nohup python3 scripts/vast/fetch_model.py "$PREFETCH" "models/$(basename "$PREFETCH")" > outputs/prefetch.log 2>&1 &
  echo "đang tải nền $PREFETCH → outputs/prefetch.log"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)"
python3 -m pytest -q
