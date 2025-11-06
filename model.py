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

# Configuration
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"   
MAX_SOURCE_LEN_FULL = 512
MAX_SOURCE_LEN_REDUCED = 384                   
MAX_TARGET_LEN = 128
BATCH_SIZE = 2                                  
EPOCHS = 1                                     
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load Train and split
data_files={"train": "train_clean.parquet","test": "test_clean.parquet"}
dataset = load_dataset("HallowsYves/CPSC483-data", data_files=data_files)
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

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
        model_inputs = tokenizer(
            batch["article"],
            max_length=max_source_len,
            truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["highlights"],
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
