# DataMate — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của tiểu thương (CSV / Excel / SQLite), chấm bằng Penalized
Accuracy (đúng +1, sai −1, abstain 0). Đề bài: [docs/datamate_track_info.md](docs/datamate_track_info.md).
Kiến trúc: [docs/KIEN_TRUC_PA.md](docs/KIEN_TRUC_PA.md).

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `datamate/` | Package chính: nạp + làm sạch dữ liệu, tầng tất định, agent LLM, bộ chọn đáp án, CLI `python -m datamate.run` |
| `datamate/synth/` | Sinh shop ngành mới + câu hỏi + đáp án oracle |
| `scripts/` | Chạy tầng tất định, chấm điểm, kiểm file nộp, dựng lại shots / luật / từ vựng |
| `scripts/vast/` | Cài đặt, bật vLLM và chạy agent trên GPU thuê |
| `tests/` | Test (`python -m pytest`), cần dữ liệu cuộc thi trong `data/` |
| `Dockerfile`, `predict.sh` | Image nộp bài: đọc `/data` (read-only), ghi `/result/submission.jsonl` |

Dữ liệu cuộc thi đặt ở `data/` (`data/train`, `data/public_test`), không đi kèm repo.

## Cài đặt (không cần GPU)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Linux: .venv/bin/python
.venv/Scripts/python -m pytest
```

## Chạy
```bash
# tầng tất định (không model), chấm trên train
python scripts/baseline_run.py data/train --out outputs/baseline_train.jsonl --score
# kiểm file nộp đủ id, đúng format
python scripts/check_submission.py <submission.jsonl> data/public_test/questions.jsonl
# chấm một submission với đáp án
python scripts/score.py <submission.jsonl> <answers.jsonl>
```

Agent đầy đủ (vLLM + Qwen3-14B) chạy trên GPU:
```bash
bash scripts/vast/setup.sh          # tải model, cài package, chạy test
bash scripts/vast/run.sh train 50   # 50 câu train mẫu, in PA
bash scripts/vast/run.sh public     # → outputs/public_<tag>/submission.jsonl
```
Chi tiết: [docs/RUN_VAST.md](docs/RUN_VAST.md).

## Docker (nộp bài)
```bash
docker build -t datamate .                          # cần models/Qwen3-14B trong thư mục build
docker run --gpus all --network none -v <test>:/data:ro -v <out>:/result datamate
```

## Model
**Qwen/Qwen3-14B** (~14.8B, Apache 2.0) phục vụ bằng vLLM. Tổng tham số ≤ 15B, weights công bố trước 1/10/2026,
chạy offline. Không dùng embedding model (khớp tên bằng rapidfuzz).
