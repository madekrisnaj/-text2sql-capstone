"""Uji signifikansi dan analisis galat."""
import numpy as np
import sqlglot
from sqlglot import exp
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(base_correct, ft_correct):
    """McNemar untuk hasil biner berpasangan (benar/salah per sampel)."""
    base_correct, ft_correct = np.array(base_correct, bool), np.array(ft_correct, bool)
    b = int(np.sum(base_correct & ~ft_correct))   # base benar, fine-tuned salah
    c = int(np.sum(~base_correct & ft_correct))   # base salah, fine-tuned benar
    p = mcnemar([[0, b], [c, 0]], exact=(b + c) < 25, correction=True).pvalue
    return {"base_benar_ft_salah": b, "base_salah_ft_benar": c, "p_value": p}


def bootstrap_ci(values, n_boot=2000, seed=42):
    """Confidence interval 95% untuk rata-rata (misal Execution Accuracy)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, len(arr)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(100 * lo, 2), round(100 * hi, 2)


def auto_tag(gold, pred, syntax_ok):
    """Kategori galat otomatis. Tetap periksa ulang secara manual."""
    if not pred:
        return "output_kosong"
    if not syntax_ok:
        return "gagal_eksekusi"
    try:
        g = sqlglot.parse_one(gold, read="sqlite")
        p = sqlglot.parse_one(pred, read="sqlite")
    except Exception:
        return "gagal_parse"
    aggs = lambda t: sorted(type(x).__name__ for x in t.find_all(exp.AggFunc))
    cols = lambda t: {c.name.lower() for c in t.find_all(exp.Column)}
    lits = lambda t: {str(x.this).lower() for x in t.find_all(exp.Literal)}
    if aggs(g) != aggs(p):
        return "beda_agregasi"
    if cols(g) != cols(p):
        return "beda_kolom"
    if lits(g) != lits(p):
        return "beda_nilai_literal"
    return "lainnya"
