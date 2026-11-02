"""Conservative exact entity resolution without changing IDs or using fuzzy scores."""
import unicodedata
from dataclasses import dataclass

from datamate.ir import PlanError


def entity_key(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


@dataclass(frozen=True)
class EntityBinding:
    column: str
    value: str
    matched_records: int


def resolve_entity(frame, column, requested, id_column=None, eligible_mask=None):
    if column not in frame or not isinstance(requested, str) or not requested.strip():
        raise PlanError("entity_binding", "entity name requires an available source column")
    eligible = frame if eligible_mask is None else frame.loc[eligible_mask]
    if any(v is None or not isinstance(v, str) or not v.strip() for v in eligible[column]):
        raise PlanError("entity_completeness", "eligible rows contain missing or invalid entity names")
    matches = [v for v in frame[column].dropna().unique()
               if isinstance(v, str) and entity_key(v) == entity_key(requested)]
    if len(matches) != 1:
        raise PlanError("entity_ambiguity", "entity must match one exact source label; unknown/colliding names cannot be guessed")
    selected = frame[frame[column] == matches[0]]
    if id_column:
        if id_column not in frame or selected[id_column].isna().any() or selected[id_column].nunique() != 1:
            raise PlanError("entity_ambiguity", "requested name maps to missing or multiple identifiers")
    return EntityBinding(column, matches[0], len(selected))
