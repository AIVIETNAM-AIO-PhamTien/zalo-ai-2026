import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from datamate.run import main


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = body["messages"][-1]["content"]
        if "Kết quả chạy" in last:
            content = "<think>xong</think>" + '<answer>{"type": "number", "answer": 350}</answer>'
        else:
            content = "```python\nprint(T['ban_hang']['Tiền'].sum())\n```"
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_finance_sem_baseline_answers_without_llm(tmp_path, monkeypatch):
    from datamate.llm import LLM
    def no_llm(*args, **kwargs):
        raise AssertionError("Deterministic finance must not call LLM")
    monkeypatch.setattr(LLM, "chat", no_llm)
    split = tmp_path / "split"
    biz = split / "data" / "synthetic"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text(
        "## Cột trong `sales.csv`\n- **id**: mã đơn\n- **date**: ngày bán\n"
        "- **status**: trạng thái đơn\n- **amount**: thành tiền sau giảm\n- **qty**: số lượng\n"
        "\n## Quy ước\n- **Giá vốn**: `sales.GiaVonLucBan` là giá vốn một đơn vị bán.\n", encoding="utf-8")
    (biz / "sales.csv").write_text(
        "id,date,status,amount,qty,GiaVonLucBan\nA,01/01/2025,Xong,1000,2,100\n"
        "B,02/01/2025,Đã huỷ,2000,1,100\n", encoding="utf-8")
    (split / "rulebook.md").write_text("RULES", encoding="utf-8")
    (split / "questions.jsonl").write_text(json.dumps({"id": "q", "business_id": "synthetic",
        "question": "Lãi gộp tháng 1/2025 là bao nhiêu?", "as_of_date": "2025-01-31"}), encoding="utf-8")
    out = tmp_path / "out" / "submission.jsonl"
    main(["--data", str(split), "--out", str(out), "--cache", str(tmp_path / "cache"),
          "--llm-url", "http://127.0.0.1:9/v1", "--workers", "1", "--ingest-workers", "1", "--llm-wait", "0"])
    assert json.loads(out.read_text(encoding="utf-8")) == {"id": "q", "type": "number", "answer": 800}
    assert json.loads((out.parent / "logs" / "q.json").read_text(encoding="utf-8"))["source"] == "baseline"


def test_agent_only_disables_all_automatic_answer_paths(tmp_path, monkeypatch):
    import importlib
    runner = importlib.import_module("datamate.run")
    split = tmp_path / "split"
    biz = split / "data" / "s"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text("## Cột trong `sales.csv`\n- **id**: mã đơn\n- **date**: ngày bán\n"
                                   "- **status**: trạng thái đơn\n- **amount**: thành tiền sau giảm\n", encoding="utf-8")
    (biz / "sales.csv").write_text("id,date,status,amount\nA,01/01/2025,Xong,100\n", encoding="utf-8")
    (split / "rulebook.md").write_text("RULEBOOK", encoding="utf-8")
    (split / "questions.jsonl").write_text(json.dumps({"id": "q", "business_id": "s",
        "question": "Doanh thu tháng 1/2025?", "as_of_date": "2025-01-31"}), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Automatic rule path invoked")
    def agent(task, context, llm, **kwargs):
        assert task.facts == [] and context.rulebook == "RULEBOOK"
        return {"type": "number", "answer": 7}, {}
    monkeypatch.setattr(runner, "baseline_answer", forbidden)
    monkeypatch.setattr(runner, "solve_plan", forbidden)
    monkeypatch.setattr(runner, "sem_facts", forbidden)
    monkeypatch.setattr(runner, "check_unanswerable", forbidden)
    monkeypatch.setattr(runner, "solve", agent)
    out = tmp_path / "out" / "submission.jsonl"
    argv = ["--data", str(split), "--out", str(out), "--cache", str(tmp_path / "cache"),
            "--agent-only", "--plan", "--sem-facts", "--workers", "1", "--ingest-workers", "1", "--llm-wait", "0"]
    runner.main(argv)
    assert json.loads(out.read_text(encoding="utf-8"))["answer"] == 7
    assert json.loads((out.parent / "logs" / "q.json").read_text(encoding="utf-8"))["source"] == "agent"
    manifest = json.loads((out.parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"]["baseline"] is False and manifest["mode"]["plan"] is False
    assert manifest["mode"]["sem_facts"] is False
    runner.main(argv)  # cùng code/cấu hình được resume
    with pytest.raises(SystemExit):
        runner.main(argv + ["--checked-joins"])  # không trộn hai thí nghiệm


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_end_to_end(tmp_path, server):
    split = tmp_path / "split"
    biz = split / "data" / "biz_t"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text("# biz_t", encoding="utf-8")
    (biz / "ban_hang.csv").write_text("Mã đơn,Ngày,Tiền\nO1,01/01/2025,\"100,000\"\nO2,02/01/2025,250k\n",
                                      encoding="utf-8")
    (split / "rulebook.md").write_text("RULES", encoding="utf-8")
    qs = [{"id": f"q{i}", "business_id": "biz_t", "question": "Tổng tiền?", "as_of_date": "2025-01-31"}
          for i in range(3)]
    (split / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
    out = tmp_path / "out" / "submission.jsonl"
    args = ["--data", str(split), "--out", str(out), "--cache", str(tmp_path / "cache"),
            "--llm-url", server, "--workers", "2", "--ingest-workers", "1", "--exec-timeout", "60"]
    main(args)
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(l["id"] for l in lines) == ["q0", "q1", "q2"]
    assert all(l == {"id": l["id"], "type": "number", "answer": 350} for l in lines)
    log = json.loads((tmp_path / "out" / "logs" / "q0.json").read_text(encoding="utf-8"))
    assert log["final"]["answer"] == 350
    assert "350000" in log["traces"][0][0]["out"]
    # chạy lại: không chạy lại câu đã có
    main(args)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_baseline_tier_answers_without_llm(tmp_path):
    split = tmp_path / "split"
    biz = split / "data" / "biz_b"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text(
        "## Cột trong `ban_hang.csv`\n- **Mã đơn**: mã đơn\n- **Ngày**: ngày bán\n"
        "- **TT**: trạng thái đơn\n- **Tiền**: thành tiền sau giảm\n", encoding="utf-8")
    (biz / "ban_hang.csv").write_text(
        "Mã đơn,Ngày,TT,Tiền\nO1,01/01/2025,Xong,100000\nO2,02/01/2025,Đã hủy,50000\nO3,03/01/2025,Xong,250k\n",
        encoding="utf-8")
    (split / "rulebook.md").write_text("RULES", encoding="utf-8")
    (split / "questions.jsonl").write_text(json.dumps(
        {"id": "b1", "business_id": "biz_b", "question": "Doanh thu tháng 1/2025 là bao nhiêu?",
         "as_of_date": "2025-01-31"}), encoding="utf-8")
    out = tmp_path / "o" / "submission.jsonl"
    # URL LLM không tồn tại: nếu agent bị gọi thì câu sẽ thành abstain
    main(["--data", str(split), "--out", str(out), "--cache", str(tmp_path / "c"),
          "--llm-url", "http://127.0.0.1:9/v1", "--workers", "1", "--ingest-workers", "1", "--llm-wait", "0"])
    assert json.loads(out.read_text(encoding="utf-8")) == {"id": "b1", "type": "number", "answer": 350000}
    log = json.loads((tmp_path / "o" / "logs" / "b1.json").read_text(encoding="utf-8"))
    assert log["source"] == "baseline"


def test_end_to_end_all_flags(tmp_path, server):
    split = tmp_path / "split"
    biz = split / "data" / "biz_t"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text("# biz_t", encoding="utf-8")
    (biz / "ban_hang.csv").write_text("Mã đơn,Ngày,Tiền\nO1,01/01/2025,\"100,000\"\nO2,02/01/2025,250k\n",
                                      encoding="utf-8")
    (split / "rulebook.md").write_text("| Luật | Mặc định |\n|---|---|\n| Tổng tiền | cộng cột Tiền |\n",
                                       encoding="utf-8")
    (split / "questions.jsonl").write_text(json.dumps(
        {"id": "q0", "business_id": "biz_t", "question": "Tổng tiền là bao nhiêu?", "as_of_date": "2025-01-31"}),
        encoding="utf-8")
    out = tmp_path / "out" / "submission.jsonl"
    main(["--data", str(split), "--out", str(out), "--cache", str(tmp_path / "cache"), "--llm-url", server,
          "--workers", "1", "--ingest-workers", "1", "--llm-wait", "0", "--exec-timeout", "60",
          "--reconcile", "--diverse", "--lint", "--col-hints"])
    assert json.loads(out.read_text(encoding="utf-8")) == {"id": "q0", "type": "number", "answer": 350}
    log = json.loads((tmp_path / "out" / "logs" / "q0.json").read_text(encoding="utf-8"))
    assert log["source"] == "agent" and log["reconciled"] is False


class CardOpenAI(FakeOpenAI):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert "gán vai trò" in body["messages"][-1]["content"]  # chỉ thẻ schema được gọi, câu hỏi do tầng tất định
        content = ('<think>{nháp}</think>{"ban_hang": {"Mã đơn": "order_id", "Ngày": "order_date", '
                   '"TT": "order_status", "Tiền": "line_amount"}}')
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_cards_auto_fills_missing_readme_columns(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), CardOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    split = tmp_path / "split"
    biz = split / "data" / "biz_c"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text("# Cửa hàng (không có mục cột)", encoding="utf-8")
    (biz / "ban_hang.csv").write_text(
        "Mã đơn,Ngày,TT,Tiền\nO1,01/01/2025,Xong,100000\nO2,02/01/2025,Đã hủy,50000\nO3,03/01/2025,Xong,250k\n",
        encoding="utf-8")
    (split / "rulebook.md").write_text("RULES", encoding="utf-8")
    (split / "questions.jsonl").write_text(json.dumps(
        {"id": "c1", "business_id": "biz_c", "question": "Doanh thu tháng 1/2025 là bao nhiêu?",
         "as_of_date": "2025-01-31"}), encoding="utf-8")
    out = tmp_path / "o" / "submission.jsonl"
    try:
        main(["--data", str(split), "--out", str(out), "--cache", str(tmp_path / "c"), "--cards", "auto",
              "--llm-url", f"http://127.0.0.1:{srv.server_port}/v1", "--workers", "1", "--ingest-workers", "1", "--llm-wait", "0"])
    finally:
        srv.shutdown()
    assert json.loads(out.read_text(encoding="utf-8")) == {"id": "c1", "type": "number", "answer": 350000}
    assert (tmp_path / "c" / "cards" / "biz_c.card.json").exists()
