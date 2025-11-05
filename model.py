from datasets import load_dataset
import pandas as pd

# Load Train and split
from datasets import load_dataset

data_files={"train": "train_clean.parquet","test": "test_clean.parquet"}

dataset = load_dataset("HallowsYves/CPSC483-data", data_files=data_files)

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()