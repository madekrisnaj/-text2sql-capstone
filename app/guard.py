"""Guardrail keamanan dan penyamaran PII (UU PDP)."""
import re

import pandas as pd
import sqlglot
from sqlglot import exp

# Nama ditampilkan sebagian (pseudonimisasi). Kontak ditutup penuh.
NAMA_COLUMNS = {"nama", "nama_pelanggan"}
KONTAK_COLUMNS = {"no_hp", "telepon", "email", "nik", "alamat"}
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


def samarkan_nama(v):
    """'Soleh Nashiruddin' -> 'Soleh N.'. Gelar (Dr., Hj., S.T.) dibuang. Satu kata -> 'S***'."""
    if not isinstance(v, str) or not v.strip():
        return v
    parts = [p for p in v.split(",")[0].split() if not p.endswith(".")]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1][0]}."
    return (parts[0][0] if parts else v.strip()[0]) + "***"


def _mask_value(v):
    if not isinstance(v, str):
        return v
    for pat, repl in PII_PATTERNS:
        v = pat.sub(repl, v)
    return v


def mask_pii(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        c = str(col).lower()
        if c in NAMA_COLUMNS:
            df[col] = df[col].map(samarkan_nama)
        elif c in KONTAK_COLUMNS:
            df[col] = "[DISAMARKAN]"
        elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(_mask_value)
    return df
