import os, re, json, pickle
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)

# Device selection
device = torch.device("cpu")
torch.set_num_threads(os.cpu_count())
print(f"Using device: {device}")
# Configuration
MODEL_NAME = "t5-small"  
MAX_SOURCE_LEN_FULL = 256
MAX_SOURCE_LEN_REDUCED = 192
MAX_TARGET_LEN = 64
BATCH_SIZE = 1
EPOCHS = 1
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load Train and split
data_files={"train": "train_clean.parquet","test": "test_clean.parquet"}
dataset = load_dataset("HallowsYves/CPSC483-data", data_files=data_files)
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

MAX_TRAIN, MAX_TEST = 2000, 500
if len(train_df) > MAX_TRAIN:
    train_df = train_df.sample(n=MAX_TRAIN, random_state=SEED)
if len(test_df) > MAX_TEST:
    test_df = test_df.sample(n=MAX_TEST, random_state=SEED)

# Reduced-feature 
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def lead3(text: str, k: int = 3) -> str:
    """
        Checks if the input is a string
        Splits the article into sentences, based off of . ! ? or whitespace 
        Keeps the first 3 sentences
    """
    if not isinstance(text, str):
        return ""
    sents = _SENT_SPLIT.split(text.strip())
    return " ".join(sents[:k])

train_df_reduced = train_df.copy()
test_df_reduced = test_df.copy()

train_df_reduced["article"] = train_df_reduced["article"].apply(lead3)
test_df_reduced["article"] = test_df_reduced["article"].apply(lead3)

# Tokenizer & preprocessing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def make_preprocess(max_source_len):
    def _prep(batch):
        inputs = ["summarize: " + x for x in batch["article"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_len,
            truncation=True
        )
        labels = tokenizer(
            text_target=batch["highlights"],
            max_length=MAX_TARGET_LEN,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _prep


# Tokenize & HF Datasets
ds_full_train = Dataset.from_pandas(train_df[["article", "highlights"]])
ds_full_test = Dataset.from_pandas(test_df[["article", "highlights"]])

ds_reduced_train = Dataset.from_pandas(train_df_reduced[["article", "highlights"]])
ds_reduced_test = Dataset.from_pandas(test_df_reduced[["article","highlights"]])

ds_full_train_tokenized = ds_full_train.map(make_preprocess(MAX_SOURCE_LEN_FULL), batched=True, remove_columns=["article", "highlights"])
ds_full_test_tokenized = ds_full_test.map(make_preprocess(MAX_SOURCE_LEN_FULL), batched=True, remove_columns=["article", "highlights"])

ds_reduced_train_tokenized = ds_reduced_train.map(make_preprocess(MAX_SOURCE_LEN_REDUCED), batched=True, remove_columns=["article", "highlights"])
ds_reduced_test_tokenized = ds_reduced_test.map(make_preprocess(MAX_SOURCE_LEN_REDUCED), batched=True, remove_columns=["article", "highlights"])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)

# Training helper

def train_and_save(train_ds, eval_ds, out_directory, note):
    output_path = Path(out_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.config.use_cache = False  
    model.gradient_checkpointing_enable()
    args = TrainingArguments(
        output_dir=str(output_path / "trainer"),
        per_device_train_batch_size=BATCH_SIZE,   
        per_device_eval_batch_size=BATCH_SIZE,   
        gradient_accumulation_steps=2,           
        num_train_epochs=EPOCHS,
        learning_rate=5e-5,
        logging_steps=50,
        seed=SEED,
        fp16=False,
        do_eval=False,
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save HF model + tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    models_directory = Path("models"); models_directory.mkdir(exist_ok=True)
    save_path = models_directory / f"finalized_{output_path.name}.sav"
    with open(save_path, "wb") as f:
        pickle.dump({
            "type": "seq2seq",
            "base": MODEL_NAME,
            "dir": str(output_path),
            "note": note
        }, f)
    return str(save_path)


sav_full = train_and_save(
    ds_full_train_tokenized,
    ds_full_test_tokenized,
    "models/m1_full_distilbart",
    note="full"
)
sav_red = train_and_save(
    ds_reduced_train_tokenized,
    ds_reduced_test_tokenized,
    "models/m2_reduced_lead3",
    note="lead3"
)

print("Saved artifacts:\n -", sav_full, "\n -", sav_red,
      "\n - models/m1_full_distilbart/\n - models/m2_reduced_lead3/")

from transformers import AutoTokenizer as _ATok, AutoModelForSeq2SeqLM as _AModel

sample_text_full = test_df["article"].iloc[0]
sample_text_red  = test_df_reduced["article"].iloc[0]

# Full model generate
tok_full = _ATok.from_pretrained("models/m1_full_distilbart")
mdl_full = _AModel.from_pretrained("models/m1_full_distilbart").to(device)
inputs = tok_full(sample_text_full, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LEN_FULL)
inputs = {k: v.to(device) for k, v in inputs.items()}
gen_ids = mdl_full.generate(**inputs, max_new_tokens=80)
print("[M1 FULL]", tok_full.decode(gen_ids[0], skip_special_tokens=True)[:300], "...")

# Reduced model generate
tok_red = _ATok.from_pretrained("models/m2_reduced_lead3")
mdl_red = _AModel.from_pretrained("models/m2_reduced_lead3").to(device)
inputs = tok_red(sample_text_red, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LEN_REDUCED)
inputs = {k: v.to(device) for k, v in inputs.items()}
gen_ids = mdl_red.generate(**inputs, max_new_tokens=80)
print("[M2 REDUCED]", tok_red.decode(gen_ids[0], skip_special_tokens=True)[:300], "...")
