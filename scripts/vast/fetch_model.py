"""Tải một model HuggingFace về thư mục cho trước (dùng trong setup.sh).

    python3 scripts/vast/fetch_model.py Qwen/Qwen3-14B models/Qwen3-14B

Danh sách file + sha256 lấy từ API HuggingFace; nội dung tải từ ModelScope trước (CDN HF hay bóp tốc độ máy vast:
đo được ~1.6 MB/s mỗi luồng so với ~20 MB/s ở ModelScope), mỗi file một luồng, kiểm sha256 file LFS; file nào
ModelScope không có hoặc lệch mã băm thì tải lại từ HuggingFace. File đã có đúng kích thước thì bỏ qua.
"""
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

repo, out = sys.argv[1], sys.argv[2]
SOURCES = [f"https://www.modelscope.cn/models/{repo}/resolve/master/{{}}", f"https://huggingface.co/{repo}/resolve/main/{{}}"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fetch(item: dict) -> str:
    path, size, oid = item["path"], item["size"], (item.get("lfs") or {}).get("oid")
    dst = os.path.join(out, path)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return f"có sẵn {path}"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    for src in SOURCES:
        try:
            with urllib.request.urlopen(src.format(path), timeout=60) as r, open(dst + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 24)
        except Exception as e:
            print(f"lỗi {path} từ {src.split('/')[2]}: {e!r}", flush=True)
            continue
        if os.path.getsize(dst + ".part") == size and (oid is None or sha256(dst + ".part") == oid):
            os.replace(dst + ".part", dst)
            return f"xong {path} ({src.split('/')[2]})"
        print(f"lệch kích thước / mã băm {path} từ {src.split('/')[2]}", flush=True)
    raise RuntimeError(f"không tải được {path}")


with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1", timeout=60) as r:
    files = [f for f in json.load(r) if f["type"] == "file"]
with ThreadPoolExecutor(8) as ex:
    for msg in ex.map(fetch, files):
        print(msg, flush=True)
print("model ->", out, flush=True)
