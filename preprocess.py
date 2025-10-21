from datasets import load_dataset
import pandas as pd

dataset = load_dataset("HallowsYves/CPSC483-data")


train_df = pd.DataFrame(dataset["train"])


# TODO: Clean up Datasets