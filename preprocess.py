from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
from datasets import load_dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split

dataset = load_dataset("HallowsYves/CPSC483-data")
train_df = dataset['train'].to_pandas()
print("imported data from hugging face. \n")


hf_logging.set_verbosity_error() 

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\(CNN\).*?-', '', text)
    text = re.sub(r'\[[0-9]+\]', '', text)
    text = text.strip()
    return text

# Drop ID Column
train_df = train_df[['article', 'highlights']]

# Cleaning up text in the Article and Highlights columns
train_df['article'] = train_df['article'].apply(clean_text)
train_df['highlights'] = train_df['highlights'].apply(clean_text)

# Trim passages, so that they meet token limits
tokenizer = AutoTokenizer.from_pretrained(
    "google/pegasus-cnn_dailymail",
    use_fast=True,
    model_max_length=10**9
)

def token_length(text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

def row_within_length(row, max_input=1024, max_output=128):
    return token_length(row['article']) <= max_input and token_length(row['highlights']) <= max_output

print("Tokenizing")
mask = train_df.apply(row_within_length, axis=1)
filtered_train_df = train_df[mask].reset_index(drop=True)
print(f"Kept {len(filtered_train_df)} / {len(train_df)} rows")

print("Creating train/test split (80/20)...")
train_split, test_split = train_test_split(
    filtered_train_df, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train rows: {len(train_split)} | Test rows: {len(test_split)}")

print("Exporting splits as parquet files: 'train_clean.parquet' and 'test_clean.parquet'")
train_split.to_parquet("train_clean.parquet", index=False)
test_split.to_parquet("test_clean.parquet", index=False)