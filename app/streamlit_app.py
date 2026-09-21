"""Database Copilot: aplikasi demo Text-to-SQL berbasis Streamlit.

Fitur:
- Model GGUF dimuat langsung (default) atau lewat Hugging Face Space (isi SPACE_ID di Secrets).
- Guardrail read-only: hanya satu query SELECT yang boleh dijalankan.
- Perbaikan otomatis berbasis eksekusi: jika query error, pesan error dikirim
  kembali ke model untuk diperbaiki (maksimal MAX_PERBAIKAN kali).
- Penyamaran PII pada hasil query (UU PDP).
"""
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from guard import is_safe, mask_pii

HERE = Path(__file__).parent
DB_PATH = HERE / "ecommerce.db"
SYS = "You are a SQLite expert. Return exactly one SQL query, no explanation."
MAX_PERBAIKAN = 2
CONTOH = [
    "Berapa jumlah pelanggan yang tinggal di Jakarta?",
    "Berapa total pendapatan untuk setiap kategori barang?",
    "Siapa 5 pelanggan dengan total belanja terbesar?",
    "Tampilkan nama barang yang belum pernah terjual.",
    "Hapus semua data pelanggan.",
]


def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return default


MODEL_REPO = get_secret("MODEL_REPO", "Kriskris28/qwen15-sql-gguf")
MODEL_FILE = get_secret("MODEL_FILE", "q15-sql-q4_k_m.gguf")
SPACE_ID = get_secret("SPACE_ID", "")


@st.cache_resource(show_spinner="Memuat model. Proses pertama sekitar 1-3 menit...")
def load_local_model():
    from llama_cpp import Llama
    return Llama.from_pretrained(repo_id=MODEL_REPO, filename=MODEL_FILE,
                                 n_ctx=2048, n_threads=2, verbose=False)


@st.cache_resource(show_spinner="Menghubungkan ke backend...")
def load_space_client():
    from gradio_client import Client
    return Client(SPACE_ID)


@st.cache_data
def load_schema():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        con.close()
    return "\n".join(r[0] for r in rows)


def clean_sql(text):
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1)
    return t.strip().rstrip(";").strip()


def generate_sql(question, schema, prev_sql=None, error=None):
    """Buat SQL. Jika prev_sql dan error diisi, model diminta memperbaiki query sebelumnya."""
    instruksi_perbaikan = (f"The previous query failed with error: {error}. "
                           "Fix it using only the tables and columns in the schema. "
                           "Use table aliases consistently. Return only the corrected SQL.")
    if SPACE_ID:
        q = question if not prev_sql else f"{question}\nPrevious query: {prev_sql}\n{instruksi_perbaikan}"
        return clean_sql(load_space_client().predict(q, schema, api_name="/predict"))

    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"}]
    if prev_sql:
        msgs += [{"role": "assistant", "content": prev_sql},
                 {"role": "user", "content": instruksi_perbaikan}]
    out = load_local_model().create_chat_completion(messages=msgs, temperature=0, max_tokens=200)
    return clean_sql(out["choices"][0]["message"]["content"])


def execute(sql, max_rows=100, timeout=3.0):
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    deadline = time.time() + timeout
    con.set_progress_handler(lambda: time.time() > deadline, 10_000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchmany(max_rows), columns=cols)
    finally:
        con.close()


def jalankan_dengan_perbaikan(question, schema, gen_fn=generate_sql, max_perbaikan=MAX_PERBAIKAN):
    """Generate lalu eksekusi. Jika error, minta model memperbaiki.
    Mengembalikan (sql_final, hasil_df atau None, status, riwayat_gagal)."""
    riwayat = []
    sql = gen_fn(question, schema)
    for percobaan in range(max_perbaikan + 1):
        if not is_safe(sql):
            return sql, None, "ditolak", riwayat
        try:
            return sql, execute(sql), "ok", riwayat
        except Exception as e:
            riwayat.append((sql, str(e)))
            if percobaan == max_perbaikan:
                return sql, None, "gagal", riwayat
            sql = gen_fn(question, schema, prev_sql=sql, error=str(e))


# ---------------- Tampilan ----------------
st.set_page_config(page_title="Database Copilot", layout="centered")
st.title("Database Copilot")
st.caption("Ajukan pertanyaan bisnis dalam bahasa sehari-hari. Sistem membuat query SQL "
           "dan menampilkan hasilnya. Data bersifat sintetis untuk keperluan demo.")

schema = load_schema()
with st.expander("Lihat skema database"):
    st.code(schema, language="sql")

with st.expander("Lihat contoh data"):
    tabel = st.selectbox("Pilih tabel", ["pelanggan", "barang", "transaksi"])
    jumlah = execute(f"SELECT COUNT(*) AS n FROM {tabel}")["n"][0]
    st.caption(f"Tabel {tabel} berisi {jumlah} baris. Ditampilkan 20 baris pertama.")
    st.dataframe(mask_pii(execute(f"SELECT * FROM {tabel} LIMIT 20")))

pilihan = st.selectbox("Contoh pertanyaan (opsional)", [""] + CONTOH)
question = st.text_input("Pertanyaan Anda", value=pilihan)

if st.button("Jalankan", type="primary") and question.strip():
    t0 = time.time()
    try:
        sql, hasil, status, riwayat = jalankan_dengan_perbaikan(question.strip(), schema)
    except Exception as e:
        st.error(f"Model gagal merespons: {e}")
        st.stop()

    if riwayat:
        with st.expander(f"Perbaikan otomatis: {len(riwayat)} query gagal sebelum hasil akhir"):
            for i, (s, err) in enumerate(riwayat, 1):
                st.markdown(f"**Percobaan {i}** gagal: `{err}`")
                st.code(s, language="sql")

    st.subheader("Query SQL")
    st.code(sql or "(kosong)", language="sql")

    if status == "ditolak":
        st.error("Ditolak oleh guardrail. Sistem hanya menjalankan satu query SELECT (read-only).")
    elif status == "gagal":
        st.warning(f"Query tetap gagal setelah {MAX_PERBAIKAN} kali perbaikan otomatis: {riwayat[-1][1]}")
    else:
        st.subheader("Hasil")
        st.dataframe(mask_pii(hasil))
        st.caption("Kolom berisi data pribadi (nama, nomor HP, email) disamarkan sesuai prinsip UU PDP.")
    st.caption(f"Waktu proses: {time.time() - t0:.1f} detik")
