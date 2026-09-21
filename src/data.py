"""Memuat dataset, menghapus duplikat, dan membagi data per skema.

Split per skema mencegah kebocoran data: satu skema tabel
tidak pernah muncul di dua split berbeda.
"""
import hashlib
import random

from datasets import load_dataset

from . import config as C


def load_splits(seed=C.SEED, n_train=C.N_TRAIN):
    df = load_dataset(C.DATASET, split="train").to_pandas()
    n_awal = len(df)
    df = df.drop_duplicates(["question", "context"]).reset_index(drop=True)
    print(f"Data awal: {n_awal:,} | setelah hapus duplikat: {len(df):,}")

    df["sid"] = df["context"].map(lambda s: hashlib.md5(s.encode()).hexdigest())
    sids = sorted(df["sid"].unique())
    random.Random(seed).shuffle(sids)
    n = len(sids)
    test_ids = set(sids[: int(0.10 * n)])
    val_ids = set(sids[int(0.10 * n): int(0.15 * n)])

    test = df[df.sid.isin(test_ids)].reset_index(drop=True)
    val = df[df.sid.isin(val_ids)].reset_index(drop=True)
    train_full = df[~df.sid.isin(test_ids | val_ids)]
    train = train_full.sample(min(n_train, len(train_full)), random_state=seed).reset_index(drop=True)
    return train, val, test


def check_leakage(train, val, test):
    """Mengembalikan jumlah skema yang bocor antar split. Hasil harus 0."""
    tr, va, te = set(train.sid), set(val.sid), set(test.sid)
    return {"train-val": len(tr & va), "train-test": len(tr & te), "val-test": len(va & te)}


def query_type_stats(df):
    """Statistik jenis query untuk Section 2 laporan."""
    a = df["answer"].str.upper()
    pct = lambda mask: round(100 * mask.mean(), 2)
    return {
        "jumlah_sampel": len(df),
        "jumlah_skema": df["sid"].nunique(),
        "join_%": pct(a.str.contains(r"\bJOIN\b")),
        "group_by_%": pct(a.str.contains(r"\bGROUP BY\b")),
        "order_by_%": pct(a.str.contains(r"\bORDER BY\b")),
        "agregasi_%": pct(a.str.contains(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(")),
        "rata2_kata_pertanyaan": round(df["question"].str.split().str.len().mean(), 1),
    }


def has_join(sql):
    return " JOIN " in f" {sql.upper()} "
