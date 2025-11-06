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
