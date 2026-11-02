"""Lossless source snapshots. Inferred headers are candidates, never destructive edits."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

LOGIC_VERSION = "foundation-2"


@dataclass(frozen=True)
class Source:
    path: str
    partition: str
    sha256: str
    encoding: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RawTable:
    name: str
    source: Source
    rows: list[tuple[Any, ...]]
    columns: tuple[str, ...] | None = None
    header_candidates: tuple[int, ...] = ()  # zero-based physical rows
    ambiguities: list[str] = field(default_factory=list)
    row_numbers: tuple[int, ...] = ()

    def header_labels(self, header: int | tuple[int, ...]) -> tuple[str, ...]:
        indices = (header,) if isinstance(header, int) and not isinstance(header, bool) else header
        if (not isinstance(indices, tuple) or not indices or self.columns is not None
                or any(type(i) is not int or not 0 <= i < len(self.rows) for i in indices)
                or indices != tuple(range(indices[0], indices[-1] + 1))):
            raise ValueError("header rows must be a contiguous ordered range")
        width = max(len(self.rows[i]) for i in indices)
        cells = {(i, j): v for i in indices for j, v in enumerate(self.rows[i])}
        # Expand only explicitly recorded Excel merges, only in header projection.
        # No value from the data body is ever filled down or horizontally.
        for region in self.source.metadata.get("merged_cells", []):
            left, top, right, bottom = openpyxl.utils.cell.range_boundaries(region)
            if top - 1 not in indices:
                continue
            value = cells.get((top - 1, left - 1))
            for i in indices:
                if top - 1 <= i < bottom:
                    for j in range(left - 1, right):
                        cells[i, j] = value
        labels = []
        for j in range(width):
            parts = []
            for i in indices:
                value = cells.get((i, j))
                if value is not None and str(value) and (not parts or parts[-1] != str(value)):
                    parts.append(str(value))
            labels.append(" / ".join(parts))
        return tuple(labels)

    def header_options(self) -> list[int | tuple[int, ...]]:
        options = list(self.header_candidates)
        for start in range(min(12, len(self.rows))):
            for size in (2, 3):
                if start + size <= min(12, len(self.rows)):
                    indices = tuple(range(start, start + size))
                    if all(any(v is not None and str(v).strip() for v in self.rows[i]) for i in indices):
                        options.append(indices)
        return options

    def project(self, header: int | tuple[int, ...] | None = None) -> pd.DataFrame:
        """Explicit projection; raw rows, empty strings, NULLs and duplicates survive.

        No header argument means positional columns (or declared SQL columns).
        Physical row indices are in attrs, not injected into user column names.
        """
        width = max((len(r) for r in self.rows), default=len(self.columns or ()))
        selected = self.header_labels(header) if header is not None else None
        start = 0 if header is None else (header if isinstance(header, int) else header[-1]) + 1
        original = self.columns if self.columns is not None else (
            selected if header is not None else tuple(f"column_{i}" for i in range(width)))
        names, used = [], set()
        for i in range(width):
            value = original[i] if i < len(original) else None
            base = str(value) if value is not None and str(value) else f"column_{i}"
            name, n = base, 1
            while name in used:
                name = f"{base}__{n}"
                n += 1
            used.add(name)
            names.append(name)
        frame = pd.DataFrame([list(r) + [None] * (width - len(r)) for r in self.rows[start:]],
                             columns=names, dtype=object)
        physical = self.row_numbers or tuple(range(1, len(self.rows) + 1))
        frame.attrs.update(source=self.source, physical_rows=list(physical[start:]),
                           original_columns=list(original), header=header, raw_table=self.name,
                           header_physical_rows=[] if header is None else [physical[i] for i in
                               ((header,) if isinstance(header, int) else header)])
        return frame

    def cell(self, physical_row: int, column: int) -> dict:
        physical = self.row_numbers or tuple(range(1, len(self.rows) + 1))
        if physical_row not in physical:
            raise IndexError("source row outside table")
        row = self.rows[physical.index(physical_row)]
        if not 0 <= column < len(row):
            raise IndexError("source cell outside table")
        return {"path": self.source.path, "partition": self.source.partition,
                "row": physical_row, "column": column, "value": row[column],
                "source_sha256": self.source.sha256}


@dataclass
class Foundation:
    tables: dict[str, RawTable]
    documents: dict[str, str]
    cache_key: str
    ambiguities: list[str] = field(default_factory=list)
    raw_files: dict[str, bytes] = field(default_factory=dict)


def candidates(rows: list[tuple]) -> tuple[int, ...]:
    # Evidence only. A row of strings can equally be a data record.
    return tuple(i for i, row in enumerate(rows[:12])
                 if len([v for v in row if v is not None and v != ""]) >= 2
                 and all(isinstance(v, str) for v in row if v is not None and v != ""))


def load_foundation(directory: str | Path, rulebook: str = "") -> Foundation:
    root = Path(directory).resolve()
    tables, documents, ambiguities, raw_files = {}, {"$rulebook": rulebook}, [], {}
    digest = hashlib.sha256((LOGIC_VERSION + "\0" + rulebook).encode())
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xlsm", ".sqlite", ".db", ".md", ".txt"}:
            continue
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        raw_files[relative] = content  # preserves Excel XML serials/formats and original encoding bytes
        sha = hashlib.sha256(content).hexdigest()
        digest.update(json.dumps([relative, sha]).encode())
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            documents[relative] = content.decode("utf-8-sig", errors="strict")
        elif suffix == ".csv":
            encoding = "utf-8-sig"
            try:
                text = content.decode(encoding)
            except UnicodeDecodeError:
                encoding = "cp1258"
                text = content.decode(encoding, errors="strict")
                ambiguities.append(f"{relative}: inferred encoding cp1258; confirm locale")
            try:
                dialect = csv.Sniffer().sniff(text[:16384], delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(io.StringIO(text, newline=""), dialect)
            rows, row_numbers, previous_line = [], [], 0
            for row in reader:
                rows.append(tuple(row))
                row_numbers.append(previous_line + 1)
                previous_line = reader.line_num
            source = Source(str(path), relative, sha, encoding, {"delimiter": dialect.delimiter})
            tables[relative] = RawTable(relative, source, rows, header_candidates=candidates(rows),
                                       ambiguities=["header requires grounding", "numeric locale unspecified"],
                                       row_numbers=tuple(row_numbers))
        elif suffix in {".xlsx", ".xlsm"}:
            wb = openpyxl.load_workbook(io.BytesIO(content), data_only=False)
            cached = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
            try:
                for ws in wb.worksheets:
                    name = f"{relative}::{ws.title}"
                    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
                    metadata = {"merged_cells": [str(r) for r in ws.merged_cells.ranges],
                                "epoch": wb.epoch.isoformat(),
                                "cell_formats": {c.coordinate: c.number_format for row in ws for c in row
                                                 if c.value is not None and c.number_format != "General"},
                                "formula_cache": {c.coordinate: cached[ws.title][c.coordinate].value
                                                  for row in ws for c in row if c.data_type == "f"}}
                    tables[name] = RawTable(name, Source(str(path), ws.title, sha, metadata=metadata), rows,
                                           header_candidates=candidates(rows),
                                           ambiguities=["header requires grounding"])
            finally:
                wb.close()
                cached.close()
        else:
            con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                    quoted = '"' + table.replace('"', '""') + '"'
                    cursor = con.execute(f"SELECT * FROM {quoted}")
                    name = f"{relative}::{table}"
                    tables[name] = RawTable(name, Source(str(path), table, sha,
                        metadata={"row_index": "snapshot ordinal; SQL has no physical row order"}),
                        [tuple(r) for r in cursor.fetchall()], tuple(d[0] for d in cursor.description))
            finally:
                con.close()
    return Foundation(tables, documents, digest.hexdigest(), ambiguities, raw_files)


@dataclass(frozen=True)
class Conversion:
    raw: Any
    value: Any
    rule: str
    error: str | None = None


def decimal_value(raw: Any, locale: str | None = None, precision: int = 38, scale: int = 9) -> Conversion:
    """Exact conversion with explicit locale and representability checks; never strip units."""
    rule = f"decimal({precision},{scale});locale={locale}"
    if not 0 <= scale <= precision <= 38:
        raise ValueError("invalid decimal precision/scale")
    if raw is None:
        return Conversion(raw, None, rule)
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float, Decimal)):
        return Conversion(raw, None, rule, "not a numeric scalar")
    text = str(raw).strip()
    if locale not in {None, "vi", "en"}:
        raise ValueError("unsupported locale")
    if isinstance(raw, str) and locale:
        group, decimal = (".", ",") if locale == "vi" else (",", ".")
        pattern = rf"[+-]?(?:\d+|\d{{1,3}}(?:{re.escape(group)}\d{{3}})+)(?:{re.escape(decimal)}\d+)?"
        if not re.fullmatch(pattern, text):
            return Conversion(raw, None, rule, "invalid grouping or numeric syntax")
        text = text.replace(group, "").replace(decimal, ".")
    elif isinstance(raw, str) and ("," in text or re.fullmatch(r"[+-]?\d{1,3}\.\d{3}", text)):
        return Conversion(raw, None, rule, "ambiguous locale")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return Conversion(raw, None, rule, "invalid numeric syntax")
    try:
        value = Decimal(text)
        digits, exponent = value.as_tuple().digits, value.as_tuple().exponent
        fractional = max(0, -exponent)
        # Trailing zeroes don't require precision, but use no context-sensitive normalize().
        while fractional and digits and digits[-1] == 0:
            digits, fractional = digits[:-1], fractional - 1
        integer = max(0, value.adjusted() + 1) if value else 0
        if fractional > scale or integer > precision - scale:
            return Conversion(raw, None, rule, "not exactly representable")
        return Conversion(raw, value, rule)
    except InvalidOperation:
        return Conversion(raw, None, rule, "invalid decimal")


def date_value(raw: Any, *, format: str | None = None, excel_epoch: dt.datetime | None = None) -> Conversion:
    rule = f"date;format={format};epoch={excel_epoch}"
    try:
        if raw is None or isinstance(raw, (dt.datetime, dt.date)):
            return Conversion(raw, raw, rule)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and excel_epoch is not None:
            value = openpyxl.utils.datetime.from_excel(raw, epoch=excel_epoch)
        elif isinstance(raw, str):
            text = unicodedata.normalize("NFC", raw).strip()
            if format == "vi_named":
                match = re.fullmatch(r"(?:ngày\s+)?(\d{1,2})\s+(?:thg\.?|tháng|thang)\s+(\d{1,2})\s+(?:năm\s+|nam\s+)?(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?", text, re.IGNORECASE)
                if not match:
                    raise ValueError("Vietnamese named-month date syntax required")
                day, month, year, hour, minute, second = match.groups()
                value = dt.datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0))
            elif format in {"dayfirst_mixed", "monthfirst_mixed"}:
                numeric = "%d/%m/%Y" if format == "dayfirst_mixed" else "%m/%d/%Y"
                parsed = [date_value(text, format=candidate) for candidate in (None, "vi_named", numeric)]
                values = [c.value for c in parsed if not c.error]
                if not values or any(v != values[0] for v in values[1:]):
                    raise ValueError("mixed date has no unique authorized conversion")
                value = values[0]
            else:
                value = dt.datetime.strptime(text, format) if format else dt.datetime.fromisoformat(text)
        else:
            raise ValueError("date format or Excel epoch required")
        return Conversion(raw, value, rule)
    except (ValueError, TypeError, OverflowError) as exc:
        return Conversion(raw, None, rule, str(exc))
