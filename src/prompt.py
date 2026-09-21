"""Format prompt ChatML. Dipakai sama persis untuk training, evaluasi, dan aplikasi."""

SYS = "You are a SQLite expert. Return exactly one SQL query, no explanation."
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def build_prompt(context, question):
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Schema:\n{context}\n\nQuestion: {question}"},
    ]


def to_train_text(row, tok):
    """Teks lengkap (prompt + jawaban) untuk SFT."""
    msgs = build_prompt(row.context, row.question)
    msgs.append({"role": "assistant", "content": row.answer})
    return tok.apply_chat_template(msgs, tokenize=False)
