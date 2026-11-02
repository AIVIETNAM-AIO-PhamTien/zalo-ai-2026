import json

from datamate.agent import ShopContext, Task, build_messages
from datamate.shots import ShotBank, format_shots

SHOTS = [
    {"id": "q1", "biz": "biz_a", "question": "Doanh thu tháng 5/2025 là bao nhiêu?", "code": "x = 1", "answer": {"type": "number", "answer": 1}},
    {"id": "q2", "biz": "biz_b", "question": "Doanh thu tháng 3/2026 là bao nhiêu?", "code": "x = 2", "answer": {"type": "number", "answer": 2}},
    {"id": "q3", "biz": "biz_c", "question": "Top 3 khách mua nhiều nhất tuần rồi?", "code": "x = 3", "answer": {"type": "list", "answer": ["A"]}},
]


def _bank(tmp_path):
    p = tmp_path / "shots.jsonl"
    p.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in SHOTS), encoding="utf-8")
    return ShotBank(p)


def test_find_skips_same_shop_and_same_question(tmp_path):
    bank = _bank(tmp_path)
    hits = bank.find("doanh thu thang 7/2025 la bao nhieu", "biz_a", k=3, exclude_id="q9")
    assert [s["id"] for s in hits] == ["q2"]  # q1 cùng shop; q3 khác mẫu câu
    assert bank.find("Doanh thu tháng 3/2026 là bao nhiêu?", "biz_x", exclude_id="q2")[0]["id"] == "q1"


def test_shots_go_after_brief_and_before_question(tmp_path):
    import datetime as dt
    ctx = ShopContext(cache_path=tmp_path / "x.pkl", calendar_csv=None, readme="R", brief="BRIEF",
                      rulebook="RB", shots=_bank(tmp_path))
    user = build_messages(Task("q9", "biz_z", "Doanh thu tháng 1/2026 là bao nhiêu?", dt.date(2026, 2, 1)), ctx)[1]["content"]
    assert user.index("BRIEF") < user.index("Ví dụ 1") < user.index("# Câu hỏi")
    assert "x = 1" in user and format_shots([]) == ""
