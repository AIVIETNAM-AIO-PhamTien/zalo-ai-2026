"""Independent ledger capabilities; source selection is by documented semantics."""
from datamate.line_ledger import LineLedger
from datamate.service_ledger import ServiceLedger


def prepare(shop, rulebook=''):
    result = []
    for cls in (LineLedger, ServiceLedger):
        try:
            view = cls.from_shop(shop, rulebook)
        except (ValueError, TypeError, KeyError, AttributeError):
            view = None
        if view is not None:
            result.append(view)
    return result


def answer(question, as_of, views, periods):
    for view in views:
        value = view.answer(question, as_of, periods)
        if value is not None:
            return value
    return None
