"""Execute supported temporal rulebook clauses with exact source evidence."""
from datamate.ir import PlanError
from datamate.parsing import fold
from datamate.semantics import Evidence
from datamate.temporal import PeriodPolicy, resolve_window


SUPPORTED_PERIODS = {
    "Tuần": "bắt đầu thứ Hai. Tuần này: từ thứ Hai của tuần chứa hôm nay đến hết hôm nay. Tuần trước, tuần rồi: cả tuần liền trước, thứ Hai đến Chủ nhật",
    "Tháng này": "từ ngày 1 đến hết hôm nay. So với tháng trước thì lấy cùng số ngày của tháng trước: từ ngày 1 đến hết ngày trùng số với hôm nay; tháng trước không có ngày đó thì đến hết tháng trước",
    "Từ đầu năm, từ đầu năm tới nay": "từ 01/01 của năm nay đến hết hôm nay",
}


def grounded_window(phrase, reference, documents):
    try:
        return resolve_window(phrase, reference), []
    except PlanError as error:
        if error.code != "temporal_policy":
            raise
    normalized = fold(phrase)
    if normalized in {"tuan nay", "tuan truoc"}:
        label = "Tuần"
    elif normalized in {"thang nay", "thang truoc", "cung ky thang truoc"}:
        label = "Tháng này"
    elif normalized == "tu dau nam":
        label = "Từ đầu năm, từ đầu năm tới nay"
    else:
        raise PlanError("temporal_unsupported", "no supported temporal rule adapter for full phrase")
    text = documents.get("$rulebook", "")
    matches, offset = [], 0
    for line in text.splitlines(keepends=True):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells and fold(cells[0]) == fold(label):
            if len(cells) != 2 or fold(cells[1]) != fold(SUPPORTED_PERIODS[label]):
                raise PlanError("temporal_policy", "rulebook period semantics changed or unsupported: " + label)
            matches.append(Evidence("$rulebook", offset, offset + len(line), line))
        offset += len(line)
    if len(matches) != 1:
        raise PlanError("temporal_policy", "one complete applicable period policy rule required: " + label)
    policy = PeriodPolicy(0, True, label == "Tháng này", matches[0].quote)
    return resolve_window(phrase, reference, policy), matches
