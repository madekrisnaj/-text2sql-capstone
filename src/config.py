"""Konfigurasi terpusat proyek Text-to-SQL.

Semua path, nama model, dan hyperparameter diatur di sini.
Jangan menulis path di tempat lain agar kode tetap reproducible.
"""
import os
from pathlib import Path

# Username Hugging Face diambil dari Colab Secrets (HF_USERNAME)
HF_USERNAME = os.environ.get("HF_USERNAME", "GANTI_USERNAME_HF")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "b-mc2/sql-create-context"
NIM_MODEL = "meta/llama-3.3-70b-instruct"
SEED = 42

# Ukuran data
N_TRAIN = 15000      # subsample data latih sesuai panduan
N_TEST_EVAL = 1000   # sampel test untuk evaluasi model HF
N_NIM_EVAL = 200     # sampel test untuk baseline NIM (rate limit)
N_GGUF_EVAL = 150    # sampel test untuk evaluasi ulang GGUF di CPU

# Repository Hugging Face Hub
REPO_ADAPTER = f"{HF_USERNAME}/qwen15-sql-lora"
REPO_GGUF = f"{HF_USERNAME}/qwen15-sql-gguf"
GGUF_FILE = "q15-sql-q4_k_m.gguf"

# Lokasi output: Google Drive jika ter-mount (Jalur A), folder lokal jika tidak (Jalur B)
ROOT = Path(__file__).resolve().parents[1]
DRIVE = Path("/content/drive/MyDrive/text2sql-capstone")
OUT = DRIVE if Path("/content/drive/MyDrive").exists() else ROOT / "outputs"
DATA_DIR = OUT / "data"
CKPT_DIR = OUT / "checkpoints"
RESULT_DIR = OUT / "results"
for _d in (DATA_DIR, CKPT_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CASE_STUDY_FILE = ROOT / "data" / "case_study_id.jsonl"
ECOMMERCE_DB = ROOT / "app" / "ecommerce.db"

# Konfigurasi QLoRA (T4-safe, sesuai panduan resmi)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
TRAIN = dict(epochs=1, batch_size=8, grad_accum=2, lr=2e-4, max_seq_length=512)
