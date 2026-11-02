"""Experimental offline CPU/local-model runner: python -m datamate.predict_v2.

Only question fields and business sources enter inference. Gold is never opened.
Legacy remains default until actual validation supports promoting this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
import uuid
from pathlib import Path

from datamate.model import LocalModel
from datamate.pipeline import BusinessCache, Checkpoint, atomic_json, run_questions
from datamate.policy import Budget


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path):
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("each question must be a JSON object")
        for field in ("id", "business_id", "question"):
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                raise ValueError("question requires nonempty id/business_id/question")
        business = raw["business_id"]
        if business in {".", ".."} or any(c in business for c in "/\\:"):
            raise ValueError("business_id must name one source directory")
        if raw["id"] in seen:
            raise ValueError("duplicate question ID")
        seen.add(raw["id"])
        rows.append({k: raw[k] for k in ("id", "business_id", "question", "as_of_date") if k in raw})
    return rows


def source_manifest(data, businesses):
    sources = {}
    for business in businesses:
        root = (data / business).resolve()
        if not root.is_relative_to(data.resolve()):
            raise ValueError("business source escapes data directory")
        if not root.is_dir():
            sources[business] = {"missing": True}
            continue
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xlsm", ".db", ".sqlite", ".md", ".txt"}:
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[business] = files
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="split with questions.jsonl, rulebook.md, data/<business>")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--business", nargs="+", help="explicit business subset; does not alter sources or semantics")
    parser.add_argument("--projections", type=Path, help="trusted JSON: business -> table -> column -> conversion spec")
    parser.add_argument("--question-seconds", type=float, default=30)
    parser.add_argument("--run-seconds", type=float, default=19800)
    parser.add_argument("--model-url")
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--parameters", type=int)
    parser.add_argument("--model-file", type=Path, help="local immutable checkpoint used by the serving process")
    parser.add_argument("--model-sha256", help="expected SHA-256 of the local checkpoint")
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.question_seconds, args.run_seconds)):
        parser.error("time budgets must be positive finite seconds")
    root, out = args.data.resolve(), args.out_dir.resolve()
    if out.is_relative_to(root):
        parser.error("output must be outside read-only input split")
    model_args = (args.model_url, args.model, args.model_revision, args.parameters)
    if any(v is not None for v in model_args) and not all(v is not None for v in model_args):
        parser.error("local model needs URL, model, pinned revision and parameter count")
    if args.model_url and (args.model_file is None or args.model_sha256 is None):
        parser.error("local model needs a checkpoint file and expected SHA-256")
    if not args.model_url and (args.model_file is not None or args.model_sha256 is not None):
        parser.error("checkpoint arguments require a local model")
    if args.model_sha256 is not None and (len(args.model_sha256) != 64 or
                                          any(c not in "0123456789abcdefABCDEF" for c in args.model_sha256)):
        parser.error("model SHA-256 must be 64 hexadecimal characters")
    started = time.monotonic()
    try:
        questions = load_questions(root / "questions.jsonl")
        parent_question_count = len(questions)
        if args.business:
            requested = set(args.business)
            if requested - {q["business_id"] for q in questions}:
                raise ValueError("requested business is absent from question split")
            questions = [q for q in questions if q["business_id"] in requested]
        rulebook = (root / "rulebook.md").read_text(encoding="utf-8-sig")
        projections = json.loads(args.projections.read_text(encoding="utf-8")) if args.projections else {}
        if not isinstance(projections, dict):
            raise ValueError("projections must be a JSON object")
        business_ids = sorted({q["business_id"] for q in questions})
        sources = source_manifest(root / "data", business_ids)
        model = LocalModel(*model_args) if args.model_url else None
        checkpoint_sha = hash_file(args.model_file.resolve()) if args.model_file else None
        if checkpoint_sha and checkpoint_sha != args.model_sha256.lower():
            raise ValueError("local model checkpoint SHA-256 differs from expected")
        manifest = {"runner": "experimental-v2", "questions_sha256": hash_json(questions),
                    "parent_question_count": parent_question_count, "selected_businesses": business_ids,
                    "source_hashes": sources, "rulebook_sha256": hashlib.sha256(rulebook.encode()).hexdigest(),
                    "projections_sha256": hash_json(projections),
                    "code_sha256": hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
                        for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest(),
                    "model": args.model, "model_revision": args.model_revision, "declared_parameters": args.parameters,
                    "model_sha256": checkpoint_sha,
                    "question_seconds": args.question_seconds, "run_seconds": args.run_seconds,
                    "dependencies": {name: importlib.metadata.version(name) for name in ("duckdb", "pandas", "openpyxl", "jsonschema")}}
        checkpoint = Checkpoint(out, [q["id"] for q in questions], manifest)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    cache = BusinessCache(max_businesses=1)
    diagnostics = []
    traces = []
    deadline = started + args.run_seconds
    for business in business_ids:
        pending = [q for q in questions if q["business_id"] == business and q["id"] not in checkpoint.rows]
        if not pending:
            continue
        prepared = None
        if time.monotonic() < deadline:
            try:
                prepared = cache.get(root / "data" / business, rulebook, projections.get(business))
                diagnostics.append({"business_id": business, "preparation": prepared.diagnostics})
            except Exception as exc:
                diagnostics.append({"business_id": business, "error_type": type(exc).__name__, "stage": "preparation"})
        for question in pending:
            remaining = deadline - time.monotonic()
            if prepared is None or remaining <= 0:
                checkpoint.record(question["id"], {"type": "abstain"})
                continue
            traces.extend(run_questions([question], prepared, checkpoint, model,
                budget=Budget(max_seconds=min(args.question_seconds, remaining))))
    summary = {"completed": len(checkpoint.rows), "total": len(questions),
               "answered": sum(r["type"] != "abstain" for r in checkpoint.rows.values()),
               "runtime_seconds_this_invocation": time.monotonic() - started,
               "llm_calls_this_invocation": sum(t.get("llm_calls", 0) for t in traces),
               "tokens_this_invocation": sum(t.get("tokens", 0) for t in traces),
               "diagnostics": diagnostics, "note": "Inference summary, not accuracy evaluation; no gold read."}
    atomic_json(out / "runs" / (uuid.uuid4().hex + ".json"), summary)
    atomic_json(out / "run_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "diagnostics"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
