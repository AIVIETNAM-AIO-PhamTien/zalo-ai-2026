import json
import sqlite3
import subprocess
import sys

import pytest

from datamate.predict_v2 import hash_file, load_questions, main


def split(tmp_path):
    root = tmp_path / "split"
    business = root / "data" / "unseen-clinic"
    business.mkdir(parents=True)
    with sqlite3.connect(business / "visits.db") as con:
        con.execute("CREATE TABLE visits (duration INTEGER, ref TEXT)")
        con.executemany("INSERT INTO visits VALUES (?,?)", [(10, "a"), (20, "b")])
    (root / "rulebook.md").write_text("Synthetic clinic; direct source operations allowed.")
    questions = [
        {"id": "sum", "business_id": "unseen-clinic", "question": "Tổng cột `duration` của tất cả bản ghi trong bảng `visits.db::visits`", "gold": "DO_NOT_READ_GOLD"},
        {"id": "ambiguous", "business_id": "unseen-clinic", "question": "Doanh thu là bao nhiêu?"},
        {"id": "missing", "business_id": "missing-business", "question": "Đếm tất cả bản ghi trong bảng `nope`"},
    ]
    (root / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions), encoding="utf-8")
    # Invalid gold file proves inference cannot depend on successfully parsing gold.
    (root / "answers.jsonl").write_bytes(b"invalid gold json\x00DO_NOT_READ_GOLD")
    return root


def test_cpu_cli_subprocess_resume_and_gold_separation(tmp_path):
    root = split(tmp_path)
    out = tmp_path / "result"
    command = [sys.executable, "-m", "datamate.predict_v2", "--data", str(root), "--out-dir", str(out)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    rows = [json.loads(l) for l in (out / "submission.jsonl").read_text().splitlines()]
    assert rows == [{"id": "sum", "type": "number", "answer": 30},
                    {"id": "ambiguous", "type": "abstain"}, {"id": "missing", "type": "abstain"}]
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["llm_calls_this_invocation"] == 0
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr
    assert len(list((out / "runs").glob("*.json"))) == 2
    for file in out.rglob("*.json*"):
        assert "DO_NOT_READ_GOLD" not in file.read_text(encoding="utf-8")
    (root / "data" / "unseen-clinic" / "README.md").write_text("changed source")
    changed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert changed.returncode != 0
    assert "differ" in changed.stderr


def test_outputs_inside_data_and_bad_budgets_rejected(tmp_path):
    root = split(tmp_path)
    with pytest.raises(SystemExit):
        main(["--data", str(root), "--out-dir", str(root / "result")])
    with pytest.raises(SystemExit):
        main(["--data", str(root), "--out-dir", str(tmp_path / "result"), "--run-seconds", "nan"])


def test_local_model_requires_verified_checkpoint(tmp_path, capsys):
    root = split(tmp_path)
    checkpoint = tmp_path / "model.gguf"
    checkpoint.write_bytes(b"pinned synthetic fixture")
    assert len(hash_file(checkpoint)) == 64
    base = ["--data", str(root), "--out-dir", str(tmp_path / "result"),
            "--model-url", "http://127.0.0.1:8000/v1", "--model", "local",
            "--model-revision", "pinned", "--parameters", "4000000000"]
    with pytest.raises(SystemExit):
        main(base)
    with pytest.raises(SystemExit):
        main(base + ["--model-file", str(checkpoint), "--model-sha256", "0" * 64])
    assert "SHA-256 differs" in capsys.readouterr().err


@pytest.mark.parametrize("business", ["../outside", "a/b", "C:\\outside", ".."])
def test_question_business_cannot_escape_source_root(tmp_path, business):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"id": "q", "business_id": business, "question": "count"}))
    with pytest.raises(ValueError):
        load_questions(path)


def test_duplicate_question_ids_rejected(tmp_path):
    path = tmp_path / "questions.jsonl"
    row = json.dumps({"id": "q", "business_id": "b", "question": "count"})
    path.write_text(row + "\n" + row)
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_non_object_question_rejected(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_questions(path)


def test_explicit_business_subset_is_recorded_and_unknown_business_rejected(tmp_path):
    root = split(tmp_path)
    out = tmp_path / "result"
    assert main(["--data", str(root), "--out-dir", str(out), "--business", "unseen-clinic"]) == 0
    state = json.loads((out / "checkpoint.json").read_text())
    assert set(state["rows"]) == {"sum", "ambiguous"}
    assert state["manifest"]["parent_question_count"] == 3
    assert state["manifest"]["selected_businesses"] == ["unseen-clinic"]
    with pytest.raises(SystemExit):
        main(["--data", str(root), "--out-dir", str(tmp_path / "invalid"), "--business", "not-in-split"])
