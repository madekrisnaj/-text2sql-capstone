"""Evaluator berbasis eksekusi SQLite.

Metode:
1. Tabel diisi baris dummy yang memuat literal dari query gold,
   sehingga klausa WHERE pada gold menghasilkan baris.
2. Query gold dan prediksi dijalankan di beberapa database dengan seed berbeda.
   Prediksi dianggap benar hanya jika hasilnya sama di semua database
   (konsep test-suite accuracy, Zhong et al., 2020).
"""
import random
import re
import sqlite3
import time
from collections import Counter

import pandas as pd
import sqlglot


def _deny_attach(action, *args):
    return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK


def seed_db(context, gold, seed, n_rows=20):
    rnd = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.executescript(context)
    lits = [a or b for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', gold)]
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", gold)
    pool = lits + nums + [str(rnd.randint(0, 100)) for _ in range(10)] + ["alpha", "beta"]
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        n_cols = len(con.execute(f'PRAGMA table_info("{t}")').fetchall())
        rows = [[rnd.choice(pool) for _ in range(n_cols)] for _ in range(n_rows)]
        con.executemany(f'INSERT INTO "{t}" VALUES ({",".join("?" * n_cols)})', rows)
    con.execute("PRAGMA query_only = ON")
    con.set_authorizer(_deny_attach)
    return con


def run_sql(con, sql, timeout=2.0):
    deadline = time.time() + timeout
    con.set_progress_handler(lambda: time.time() > deadline, 10_000)
    return con.execute(sql).fetchall()


def norm(sql):
    """Normalisasi untuk Exact Match: format ulang via sqlglot, huruf kecil."""
    sql = re.sub(r'"([^"]*)"', r"'\1'", sql)   # dataset memakai kutip ganda untuk string
    try:
        return sqlglot.transpile(sql, read="sqlite")[0].lower()
    except Exception:
        return " ".join(sql.lower().split())


def same_result(a, b, ordered):
    return a == b if ordered else Counter(a) == Counter(b)


def evaluate_pair(context, gold, pred, n_db=3, timeout=2.0):
    ordered = "ORDER BY" in gold.upper()
    res = {"gold_ok": True, "gold_empty": True, "syntax_ok": True,
           "ex": True, "em": norm(gold) == norm(pred)}
    for s in range(n_db):
        try:
            con = seed_db(context, gold, seed=s)
            g = run_sql(con, gold, timeout)
        except Exception:
            res["gold_ok"] = False
            return res
        res["gold_empty"] &= len(g) == 0
        try:
            p = run_sql(con, pred, timeout)
        except Exception:
            res["syntax_ok"], res["ex"] = False, False
            con.close()
            return res
        res["ex"] &= same_result(g, p, ordered)
        con.close()
    return res


def evaluate_frame(df, pred_col, n_db=3):
    """Evaluasi seluruh baris DataFrame (kolom: context, question, answer, pred_col)."""
    rows = [evaluate_pair(r["context"], r["answer"], r[pred_col], n_db)
            for _, r in df.iterrows()]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["pred"] = out[pred_col]
    return out


def summarize(res):
    """Ringkasan metrik. Baris dengan gold rusak dikeluarkan dari perhitungan."""
    ok = res[res["gold_ok"]]
    return {
        "n_dievaluasi": len(ok),
        "gold_rusak_%": round(100 * (1 - res["gold_ok"].mean()), 2),
        "gold_kosong_%": round(100 * ok["gold_empty"].mean(), 2),
        "syntax_validity_%": round(100 * ok["syntax_ok"].mean(), 2),
        "exact_match_%": round(100 * ok["em"].mean(), 2),
        "execution_accuracy_%": round(100 * ok["ex"].mean(), 2),
    }


def evaluate_on_db(db_path, gold, pred, timeout=3.0):
    """Evaluasi pada database nyata (studi kasus e-commerce)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        g = run_sql(con, gold, timeout)
        try:
            p = run_sql(con, pred, timeout)
        except Exception:
            return {"syntax_ok": False, "ex": False, "em": False}
        ordered = "ORDER BY" in gold.upper()
        return {"syntax_ok": True, "ex": same_result(g, p, ordered), "em": norm(gold) == norm(pred)}
    finally:
        con.close()
