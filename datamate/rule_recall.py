"""On-demand access to the same deterministic pipeline from agent Python.

The question is an input, never a lookup key. Unsupported questions stay None.
"""
from datamate.helpers import Periods
from datamate.rule_guard import Guard


class RuleRecall:
    def __init__(self, sem):
        self.sem = sem
        self._prepared = False
        self.diagnostics = {}

    def _prepare(self):
        if self._prepared:
            return
        self._prepared = True
        from datamate import auxiliary
        from datamate.baseline import prepare_sem
        from datamate.debt import DebtView
        from datamate.delivery import DeliveryView
        from datamate.lots import LotView
        from datamate.stock import StockView, base_units
        from datamate.unanswerable import ShopProfile
        sem = self.sem
        view = sem.view
        self.guard = Guard(sem._rulebook)
        self.bindings = {}
        factories = {
            'lots': lambda: LotView.from_shop(sem._shop),
            'debt': lambda: DebtView.from_shop(sem._shop),
            'delivery': lambda: DeliveryView.from_shop(sem._shop),
            'stock': lambda: StockView.from_shop(sem._shop, view.date_range()[1] if view is not None else None, base_units(view)),
            'profile': lambda: ShopProfile.from_shop(sem._shop, view.date_range()[0] if view is not None else None),
            'ledgers': lambda: auxiliary.prepare(sem._shop, sem._rulebook),
        }
        for name, build in factories.items():
            try:
                self.bindings[name] = build()
            except (ValueError, TypeError, KeyError, AttributeError):
                self.bindings[name] = None
        if view is not None and not hasattr(view, '_sem_rules'):
            prepare_sem(view, sem._shop, sem._rulebook, getattr(sem, '_calendar', None))
        from datamate.semantic import OrderView
        self.diagnostics = {'order_view': view is not None, 'order_view_reason': OrderView.why if view is None else None,
            'quantity_view': view is not None and view.items is not None,
            'rule_changes': list(self.guard.changed),
            'auxiliary_views': {k: bool(v is not None and (not isinstance(v, list) or v)) for k,v in self.bindings.items()}}

    def answer(self, question):
        self._prepare()
        if self.guard.core:
            self.diagnostics['last_status'] = 'blocked_by_changed_rules'
            return None
        from datamate.baseline import answer, lot_answer, stock_answer, debt_any, delivery_answer
        from datamate import auxiliary
        from datamate.unanswerable import check
        sem, b = self.sem, self.bindings
        P = Periods(sem._as_of, getattr(sem, '_calendar', None))
        routes = []
        if b.get('profile') is not None:
            routes.append(('profile', lambda: check(question, sem._as_of, b['profile'], getattr(sem, '_calendar', None))))
        if b.get('lots') is not None and not self.guard.lot:
            routes.append(('lots', lambda: lot_answer(question, sem._as_of, b['lots'])))
        if b.get('stock') is not None:
            routes.append(('stock', lambda: stock_answer(question, sem._as_of, b['stock'])))
        if b.get('debt') is not None and not self.guard.debt:
            routes.append(('debt', lambda: debt_any(question, sem._as_of, b['debt'], P)))
        if b.get('delivery') is not None and not self.guard.delivery:
            routes.append(('delivery', lambda: delivery_answer(question, sem._as_of, b['delivery'], P)))
        if sem.view is not None and not self.guard.view:
            routes.append(('orders', lambda: answer(question, sem._as_of, sem.view, P)))
        if b.get('ledgers'):
            routes.append(('ledgers', lambda: auxiliary.answer(question, sem._as_of, b['ledgers'], P)))
        for route, calculate in routes:
            try:
                result = calculate()
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if result is not None:
                self.diagnostics.update(last_status='solved', last_route=route)
                return result
        self.diagnostics.update(last_status='unsupported_question_or_missing_binding', last_route=None)
        return None
