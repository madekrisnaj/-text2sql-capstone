"""Fine-tuning QLoRA dengan TRL SFTTrainer (konfigurasi T4-safe)."""
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from . import config as C
from .prompt import RESPONSE_TEMPLATE, to_train_text


def train_qlora(train_df, val_df, output_dir, max_steps=-1, n_val=500):
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(C.BASE_MODEL, quantization_config=bnb,
                                                 torch_dtype=torch.float16, device_map={"": 0})
    model.config.use_cache = False

    def to_ds(df):
        return Dataset.from_dict({"text": [to_train_text(r, tok) for r in df.itertuples()]})

    train_ds = to_ds(train_df)
    val_ds = to_ds(val_df.sample(min(n_val, len(val_df)), random_state=C.SEED))

    # Loss hanya dihitung pada jawaban SQL, bukan pada skema dan pertanyaan
    collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tok)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=C.TRAIN["epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=C.TRAIN["batch_size"],
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=C.TRAIN["grad_accum"],
        learning_rate=C.TRAIN["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=C.TRAIN["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        seed=C.SEED,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, data_collator=collator,
                         peft_config=LoraConfig(task_type="CAUSAL_LM", **C.LORA))
    trainer.model.print_trainable_parameters()

    has_ckpt = any(Path(output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=has_ckpt and max_steps < 0)
    return trainer
