"""Execution provenance and conservative diagnostics, independent of answer labels."""
import ast
import json

from datamate.answer import normalize_answer, answer_key


def _literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id in {'np', 'numpy'} and node.func.attr in {'int64', 'int32', 'float64', 'float32', 'bool_'} and len(node.args) == 1 and not node.keywords:
            return _literal(node.args[0])
    if isinstance(node, ast.List):
        return [_literal(n) for n in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(n) for n in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def output_candidate(output, kind):
    """Diagnostic only: the final printed literal can still be an intermediate value.

    Do not run or evaluate agent source, accept bare text as a final answer, or
    infer a reference answer. Numpy scalar reprs are parsed structurally.
    """
    lines = str(output or '').strip().splitlines()
    if not lines or len(lines[-1]) > 10000:
        return None
    try:
        value = _literal(ast.parse(lines[-1].strip(), mode='eval').body)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if isinstance(value, tuple):
        return None
    if isinstance(value, dict) and 'type' in value:
        return normalize_answer(value)
    return normalize_answer({'type': kind, 'answer': value})


def last_execution(trace):
    return next((step for step in reversed(trace or []) if 'out' in step), None)


def output_disagreement(answer, trace):
    step = last_execution(trace)
    if not answer or answer.get('type') in {'unanswerable', 'abstain'} or not step or step.get('err'):
        return None
    printed = output_candidate(step.get('out'), answer['type'])
    if printed is None or printed.get('type') != answer.get('type'):
        return None
    if answer_key(printed) == answer_key(answer):
        return False
    # A printed unrounded monetary amount can legitimately become whole VND.
    a, b = printed.get('answer'), answer.get('answer')
    if answer['type'] == 'number' and isinstance(a, (int, float)) and isinstance(b, int) and abs(a-b) <= 0.5:
        return None
    return True


def committed_answer(trace):
    """Only the side channel from the most recent successful cell is a commitment."""
    step = last_execution(trace)
    if not step or step.get('err'):
        return None
    return normalize_answer(step.get('execution_result'))


class ResultChannel:
    """One explicitly computed result per successful executor cell."""
    def __init__(self):
        self.value = None
        self.source = None

    def reset(self):
        self.value, self.source = None, None

    def certify(self, answer):
        value = normalize_answer(answer)
        if value is None or value['type'] == 'abstain':
            raise ValueError('Invalid deterministic result')
        if self.value is not None and answer_key(value) != answer_key(self.value):
            raise ValueError('Conflicting explicit and deterministic results')
        self.value, self.source = value, 'deterministic'

    def emit(self, kind, value=None, *, reason=None):
        def builtin(v):
            if isinstance(v, dict):
                return {str(k): builtin(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [builtin(x) for x in v]
            if type(v).__module__.startswith('numpy') and hasattr(v, 'item'):
                return v.item()
            return v
        value = builtin(value)
        raw = {'type': kind, 'reason': reason or ''} if kind == 'unanswerable' else {'type': kind, 'answer': value}
        answer = normalize_answer(raw)
        if answer is None or kind == 'abstain':
            raise ValueError('emit_result: invalid answer contract')
        if self.value is not None and answer_key(self.value) != answer_key(answer):
            raise ValueError('emit_result: conflicting results in one cell')
        self.value = answer
        self.source = self.source or 'code'
        print(json.dumps(answer, ensure_ascii=False))
