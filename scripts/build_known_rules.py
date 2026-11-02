"""Ảnh chụp các dòng luật đã mã hoá (rulebook train + public) → datamate/known_rules.json cho datamate/rule_guard.py.

    python scripts/build_known_rules.py data/train/rulebook.md data/public_test/rulebook.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.rule_guard import norm, rule_table  # noqa: E402

known: dict[str, list[str]] = {}
for p in sys.argv[1:]:
    for k, v in rule_table(Path(p).read_text(encoding="utf-8")).items():
        known.setdefault(k, [])
        if norm(v) not in known[k]:
            known[k].append(norm(v))
out = Path(__file__).resolve().parents[1] / "datamate" / "known_rules.json"
out.write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(known), "luật →", out)
