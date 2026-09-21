"""Database Copilot: aplikasi demo Text-to-SQL berbasis Streamlit.

Mode 1 (default): model GGUF dimuat langsung di Streamlit Cloud.
Mode 2 (cadangan): model berjalan di Hugging Face Space, isi SPACE_ID di Secrets.
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


MODEL_REPO = get_secret("MODEL_REPO", "GANTI_USERNAME_HF/qwen15-sql-gguf")
MODEL_FILE = get_secret("MODEL_FILE", "q15-sql-q4_k_m.gguf")
SPACE_ID = get_secret("SPACE_ID", "")


@st.cache_resource(show_spinner="Memuat model. Proses pertama sekitar 1-3 menit...")
def load_local_model():
    from llama_cpp import Llama
    return Llama.from_pretrained(repo_id=MODEL_REPO, filename=MODEL_FILE,
                                 n_ctx=1024, n_threads=2, verbose=False)


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
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1)
    return t.strip().rstrip(";").strip()


def generate_sql(question, schema):
    if SPACE_ID:
        return clean_sql(load_space_client().predict(question, schema, api_name="/predict"))
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"}]
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


# ---------------- Tampilan ----------------
st.set_page_config(page_title="Database Copilot", layout="centered")
st.title("Database Copilot")
st.caption("Ajukan pertanyaan bisnis dalam bahasa sehari-hari. Sistem membuat query SQL "
           "dan menampilkan hasilnya. Data bersifat sintetis untuk keperluan demo.")

schema = load_schema()
with st.expander("Lihat skema database"):
    st.code(schema, language="sql")

pilihan = st.selectbox("Contoh pertanyaan (opsional)", [""] + CONTOH)
question = st.text_input("Pertanyaan Anda", value=pilihan)

if st.button("Jalankan", type="primary") and question.strip():
    t0 = time.time()
    try:
        sql = generate_sql(question.strip(), schema)
    except Exception as e:
        st.error(f"Model gagal merespons: {e}")
        st.stop()

    st.subheader("Query SQL")
    st.code(sql or "(kosong)", language="sql")

    if not is_safe(sql):
        st.error("Ditolak oleh guardrail. Sistem hanya menjalankan satu query SELECT (read-only).")
    else:
        try:
            hasil = mask_pii(execute(sql))
            st.subheader("Hasil")
            st.dataframe(hasil, use_container_width=True)
            st.caption("Kolom berisi data pribadi (nama, nomor HP, email) disamarkan sesuai prinsip UU PDP.")
        except Exception as e:
            st.warning(f"Query tidak dapat dieksekusi: {e}")
    st.caption(f"Waktu proses: {time.time() - t0:.1f} detik")
