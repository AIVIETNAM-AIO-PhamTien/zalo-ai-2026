"""Conservative lexical preprocessing; preserve original text and quoted entities."""
import re
from dataclasses import dataclass


ALIASES = {"hqua": "hôm qua", "hom qua": "hôm qua", "hom nay": "hôm nay", "dthu": "doanh thu",
           "la bao nhiu": "là bao nhiêu", "la bao nhieu": "là bao nhiêu",
           "cua cua hang": "của cửa hàng"}


@dataclass(frozen=True)
class QuestionText:
    original: str
    canonical: str
    replacements: tuple[tuple[str, str], ...]


def normalize_question(text):
    replacements = []
    # A quoted/backtick source/entity string is data, never an alias target.
    parts = re.split(r'(`[^`]*`|"[^"\n]*"|\x27[^\x27\n]*\x27)', text)
    for i in range(0, len(parts), 2):
        for alias, canonical in ALIASES.items():
            def replace(match):
                replacements.append((match.group(), canonical))
                return canonical
            parts[i] = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", replace, parts[i], flags=re.IGNORECASE)
    return QuestionText(text, "".join(parts), tuple(replacements))
