"""Guardrail keamanan dan penyamaran PII (UU PDP)."""
import re

import pandas as pd
import sqlglot
from sqlglot import exp

PII_COLUMNS = {"nama", "nama_pelanggan", "no_hp", "telepon", "email", "nik", "alamat"}
PII_PATTERNS = [
    (re.compile(r"\b\d{16}\b"), "[NIK]"),
    (re.compile(r"(?:\+62|62|0)8\d{7,11}"), "[NOMOR HP]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
]


def is_safe(sql):
    """Hanya izinkan tepat satu query baca (SELECT/UNION/WITH ... SELECT)."""
    try:
        stmts = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception:
        return False
    return len(stmts) == 1 and isinstance(stmts[0], exp.Query)


def _mask_value(v):
    if not isinstance(v, str):
        return v
    for pat, repl in PII_PATTERNS:
        v = pat.sub(repl, v)
    return v


def mask_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Lapis 1: samarkan kolom PII berdasarkan nama. Lapis 2: regex pada semua nilai teks."""
    df = df.copy()
    for col in df.columns:
        if str(col).lower() in PII_COLUMNS:
            df[col] = "[DISAMARKAN]"
        elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(_mask_value)
    return df
