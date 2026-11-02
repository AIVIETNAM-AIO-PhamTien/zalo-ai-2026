# Chạy DataMate trên vast.ai

Mục tiêu: sinh `submission.jsonl` cho public test (nộp lần đầu) và đo PA trên mẫu train.

## 0. SSH key (làm một lần, trên máy của bạn)
```powershell
ssh-keygen -t ed25519            # Enter hết; đã có ~/.ssh/id_ed25519 thì bỏ qua
Get-Content ~/.ssh/id_ed25519.pub
```
Dán dòng in ra vào vast.ai → **Account → SSH Keys**. Nạp tiền (Billing) trước khi thuê.

## 1. Thuê máy
Trang **Search** của vast.ai:
- Template: **vLLM** (`vastai/vllm`, CUDA 13): vLLM + PyTorch cài sẵn → `setup.sh` bỏ qua bước cài nặng (PyPI trên vast
  có lúc chỉ ~0.3 MB/s). Template có thể tự bật một server chiếm GPU: vào máy chạy `nvidia-smi`, có tiến trình giữ VRAM
  thì `pkill -f "[v]llm"` trước. Dự phòng: **PyTorch (Vast)** (CUDA ≥ 12.8).
- Disk **≥ 120 GB** (Qwen3-14B + DataMind-14B ~60 GB); không tăng được sau khi tạo máy.
- Bộ lọc: **2× RTX 5090** (giống máy chấm), disk ≥ 150 GB, RAM ≥ 64 GB, Max CUDA ≥ 12.8,
  tốc độ tải ≥ 500 Mbps (model ~30 GB), reliability ≥ 99%.
- Bấm **Rent**. Ở **Instances**, chờ trạng thái Running, bấm nút SSH (biểu tượng `>_`) để lấy lệnh dạng
  `ssh -p PORT root@HOST -L 8080:localhost:8080` → ghi lại `PORT` và `HOST`.
- Tính tiền theo giờ kể cả khi không chạy gì: xong việc thì **Destroy** (Stop vẫn tính tiền disk).

## 2. Đưa code + dữ liệu lên máy
Trên máy của bạn (PowerShell, trong `D:\AI vietnam\zalo-ai`), đóng gói code + `data/` (~300 MB; `data/` không nằm trong git):
```powershell
tar -czf ..\zalo-ai.tgz --exclude=./.venv --exclude=./outputs --exclude=./.git --exclude=__pycache__ --exclude=.pytest_cache .
scp -P PORT ..\zalo-ai.tgz root@HOST:/workspace/
```
(Git Bash không có `rsync`, nên dùng `tar` + `scp`.)

## 3. Cài đặt (trên máy vast.ai)
```bash
ssh -p PORT root@HOST
tmux new -s dm                                   # mất kết nối thì ssh lại rồi: tmux attach -t dm
mkdir -p /workspace/zalo-ai && tar -xzf /workspace/zalo-ai.tgz -C /workspace/zalo-ai
cd /workspace/zalo-ai
ls data/public_test/data | head                  # kiểm tra tên file tiếng Việt còn nguyên
bash scripts/vast/setup.sh                       # tải Qwen/Qwen3-14B, cài package, chạy test
# MODEL=Qwen/Qwen3.5-9B bash scripts/vast/setup.sh   # model thứ hai để so sánh
```

## 4. Thử nhanh trên train (có đáp án)
```bash
bash scripts/vast/run.sh train 50                # 50 câu mẫu, in PA theo type
```
Xem từng câu trong `outputs/train_<model>_n50/logs/<id>.json` (code model viết, kết quả chạy, đáp án từng mẫu).

## 5. Chạy public test để nộp
```bash
bash scripts/vast/run.sh public
```
Kết quả: `outputs/public_<model>/submission.jsonl` (script tự kiểm tra đủ 1000 id và đúng format).
Chạy bị ngắt thì chạy lại cùng lệnh: câu đã xong được giữ nguyên.

Tải về máy của bạn (PowerShell, trong `D:\AI vietnam\zalo-ai`) — cả log để phân tích lỗi:
```bash
# trên máy vast.ai
cd /workspace/zalo-ai && tar -czf /workspace/results.tgz --exclude=outputs/cache outputs
```
```powershell
scp -P PORT root@HOST:/workspace/results.tgz .
tar -xzf results.tgz
```
Rồi nộp `outputs/public_Qwen3-14B/submission.jsonl` ở mục Submission của cổng thi (tối đa 5 lần/ngày).
Tải xong mới **Destroy** máy.

## Tham số hay chỉnh (qua biến `EXTRA`)
| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--samples` / `--max-samples` / `--min-agree` | 2 / 3 / 2 | Số mẫu mỗi câu và ngưỡng đồng thuận; không đủ thì abstain |
| `--no-thinking` | tắt | Tắt chế độ thinking của Qwen (nhanh hơn, thường kém chính xác hơn) |
| `--max-turns` | 8 | Số lượt code tối đa mỗi mẫu |
| `--workers` | 24 | Số câu chạy song song |
| `--time-budget` | 19800 s | Quá hạn thì ngừng nhận câu mới, câu còn lại ghi abstain |

Ví dụ: `EXTRA="--samples 3 --min-agree 2" bash scripts/vast/run.sh train 100`.

Đổi model: `MODEL_PATH=models/Qwen3.5-9B bash scripts/vast/run.sh public` (dừng vLLM cũ trước: `pkill -f "vllm serve"`).

## Gỡ lỗi
- vLLM không lên: xem `outputs/vllm.log`.
  - Mặc định `TP=2` (model chia đôi trên 2 card). Đừng dùng `DP=2` với bản bf16: 14B bf16 chiếm ~27.6 GiB,
    một card 32 GB không còn chỗ cho KV cache. Muốn mỗi card một bản thì dùng bản FP8:
    `MODEL=Qwen/Qwen3-14B-FP8 bash scripts/vast/setup.sh` rồi `MODEL_PATH=models/Qwen3-14B-FP8 DP=2 TP=1 bash scripts/vast/run.sh …`.
  - Báo lỗi `--hf-overrides` / rope scaling → `YARN=0 MAXLEN=32768 bash scripts/vast/run.sh …` (Qwen3 về context gốc 32k;
    câu dài có thể bị cắt, khi đó mẫu đó thất bại và câu dễ thành abstain).
- Prompt đầu mỗi câu ~12–17k token (rulebook + README + mô tả bảng); vì vậy mặc định mở context 64k.
- Câu nào lỗi: trường `error` hoặc `llm_error` trong log của câu đó.
