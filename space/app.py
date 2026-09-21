"""Backend cadangan di Hugging Face Space (SDK Gradio).
Dipakai hanya jika Streamlit Cloud kehabisan RAM."""
import gradio as gr
from llama_cpp import Llama

MODEL_REPO = "GANTI_USERNAME_HF/qwen15-sql-gguf"
MODEL_FILE = "q15-sql-q4_k_m.gguf"
SYS = "You are a SQLite expert. Return exactly one SQL query, no explanation."

llm = Llama.from_pretrained(repo_id=MODEL_REPO, filename=MODEL_FILE, n_ctx=1024, verbose=False)


def predict(question, schema):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"}]
    out = llm.create_chat_completion(messages=msgs, temperature=0, max_tokens=200)
    return out["choices"][0]["message"]["content"]


gr.Interface(fn=predict,
             inputs=[gr.Textbox(label="question"), gr.Textbox(label="schema", lines=6)],
             outputs=gr.Textbox(label="sql"),
             title="Text2SQL Backend").launch()
