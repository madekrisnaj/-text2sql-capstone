"""Fungsi inferensi: model Hugging Face, NVIDIA NIM, dan GGUF."""
import os
import re
import time

from tqdm.auto import tqdm


def clean_sql(text):
    """Membersihkan output model: buang teks reasoning, blok markdown, dan titik koma akhir."""
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1)
    return t.strip().rstrip(";").strip()


def load_hf_model(base, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def generate_hf(model, tok, prompts, batch_size=16, max_new_tokens=128):
    import torch

    outs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generate"):
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                 for p in prompts[i: i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [clean_sql(o) for o in outs]


def generate_nim(prompts, model, pause=1.5):
    """Generate SQL via NVIDIA NIM. Pengaturan diambil dari src/config.py."""
    from openai import OpenAI
    from . import config as C

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.environ["NIM_API_KEY"])
    kw = {"extra_body": C.NIM_EXTRA} if C.NIM_EXTRA else {}
    outs = []
    for p in tqdm(prompts, desc="NIM"):
        result = ""
        for attempt in range(3):
            try:
                r = client.chat.completions.create(model=model, messages=p, temperature=0,
                                                   max_tokens=C.NIM_MAX_TOKENS, **kw)
                result = r.choices[0].message.content
                break
            except Exception as e:
                if any(code in str(e) for code in ("404", "410", "401", "403")):
                    raise RuntimeError(f"Model {model} tidak bisa dipakai: {e}") from e
                print(f"Percobaan {attempt + 1} gagal: {e}")
                time.sleep(5 * (attempt + 1))
        outs.append(clean_sql(result))
        time.sleep(pause)
    return outs


def generate_gguf(prompts, repo, filename):
    from llama_cpp import Llama

    llm = Llama.from_pretrained(repo_id=repo, filename=filename, n_ctx=1024, verbose=False)
    outs = []
    for p in tqdm(prompts, desc="GGUF"):
        r = llm.create_chat_completion(messages=p, temperature=0, max_tokens=200)
        outs.append(clean_sql(r["choices"][0]["message"]["content"]))
    return outs
