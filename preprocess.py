from datasets import load_dataset
import pandas as pd
import re

dataset = load_dataset("HallowsYves/CPSC483-data")
train_df = dataset['train'].to_pandas()


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
print(train_df.head())