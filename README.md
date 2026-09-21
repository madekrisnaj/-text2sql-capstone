# Text-to-SQL Specialist 

Model Qwen2.5-1.5B-Instruct di-fine-tune dengan QLoRA pada dataset
[sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (CC-BY-4.0),
dievaluasi dengan Execution Accuracy berbasis SQLite, lalu di-deploy sebagai aplikasi Streamlit.

## Struktur
- `main.ipynb` : notebook utama (jalankan di Google Colab)
- `src/` : modul Python (data, prompt, training, evaluasi, guardrail)
- `app/` : aplikasi Streamlit dan database studi kasus
- `space/` : backend cadangan Hugging Face Space
- `ollama/Modelfile` : konfigurasi Ollama
- `data/case_study_id.jsonl` : 20 pertanyaan studi kasus berbahasa Indonesia

## Tautan
- Demo: (isi tautan Streamlit)
- Model: (isi tautan Hugging Face)

## Atribusi
Dataset sql-create-context oleh b-mc2, lisensi CC-BY-4.0.
Data studi kasus e-commerce bersifat sintetis (Faker id_ID).
