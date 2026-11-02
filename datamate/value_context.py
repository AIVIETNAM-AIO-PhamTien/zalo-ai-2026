"""Bounded local value retrieval for planning context, never entity proof.

CHESS indexes database values before schema selection. This adaptation uses
exact Unicode-preserving matches to avoid a model/download dependency; a hit
is retrieval evidence only, including when names collide across columns.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


class ValueIndex:
    def __init__(self, frames, *, profiles=None, max_cells=1_000_000, max_values=30_000):
        self.entries = {}
        self.by_token = defaultdict(set)
        self.truncated = False
        cells = 0
        for table, frame in frames.items():
            physical = frame.attrs.get("physical_rows", list(range(1, len(frame) + 1)))
            columns = list(frame.columns)
            if profiles and table in profiles:
                counts = profiles[table]["columns"]
                columns.sort(key=lambda col: (counts[col]["distinct"], col))
            for column in columns:
                for row_number, raw in zip(physical, frame[column]):
                    if not isinstance(raw, str) or not 5 <= len(raw) <= 80:
                        continue
                    cells += 1
                    if cells > max_cells:
                        self.truncated = True
                        return
                    value = normalized(raw)
                    if not value or len(value) > 80:
                        continue
                    key = (table, column, value)
                    if key not in self.entries:
                        if len(self.entries) >= max_values:
                            self.truncated = True
                            return
                        self.entries[key] = {"table": table, "column": column, "value": raw,
                                             "count": 0, "physical_rows": [],
                                             "status": "candidate"}
                        for token in set(re.findall(r"\w{3,}", value)):
                            self.by_token[token].add(key)
                    entry = self.entries[key]
                    entry["count"] += 1
                    if len(entry["physical_rows"]) < 2:
                        entry["physical_rows"].append(row_number)

    def search(self, question: str, *, tables=None, limit=12) -> dict:
        query = normalized(question)
        allowed = set(tables) if tables is not None else None
        words = re.findall(r"\w+", query)
        phrases = {" ".join(words[start:start + size])
                   for size in range(2, 5) for start in range(len(words) - size + 1)}
        phrases = {phrase for phrase in phrases if len(phrase) >= 6}
        keys = set()
        for token in set(re.findall(r"\w{3,}", query)):
            keys.update(self.by_token.get(token, ()))
        matches = []
        for key in keys:
            if allowed is not None and key[0] not in allowed:
                continue
            value = key[2]
            if re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", query):
                matches.append({**self.entries[key], "match": "exact_value"})
            elif any(phrase in value and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", value)
                     for phrase in phrases):
                matches.append({**self.entries[key], "match": "question_phrase_in_value"})
        matches.sort(key=lambda item: (item["match"] != "exact_value",
                                       -len(normalized(item["value"])),
                                       item["table"], item["column"], normalized(item["value"])))
        return {"matches": matches[:limit], "truncated": self.truncated or len(matches) > limit,
                "status": "retrieval candidates; duplicate names and business identity unresolved"}
