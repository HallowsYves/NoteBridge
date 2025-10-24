import matplotlib.pyplot as plt
import seaborn


from datasets import load_dataset



try:
    print("Getting Data from hugging face \n")
    dataset = load_dataset("HallowsYves/CPSC483-data")
    train_df = dataset["train"].to_pandas()
    print("Sucessfuly loaded data from hugging face \n")
except Exception:
    print("something went wrong.")

# Examine lengths of articles
train_df['article_length'] = train_df['article'].apply(len)
train_df['highlight_length'] = train_df['highlights'].apply(len)

train_df['compression_ratio'] = train_df['highlight_length'] / train_df['article_length']
filtered = train_df[train_df['compression_ratio'] < 0.5]



"""
    Compression ratio explains how much shorter is the summary compared to the article
    We are aming for around 0.2 ~ 0.05.
    1 means article and summary are the same length
    > 1 Summary > article.

"""


print("Plotting data.\n")

print("Article vs. Summary Length")
seaborn.scatterplot(x='article_length', y='highlight_length', data=train_df, alpha=0.3)
plt.title("Article vs. Summary Length")
plt.xlabel("Article Length")
plt.ylabel("Summary Length")
plt.show()


print("Distribution of Article Lengths")
seaborn.histplot(train_df['article_length'], bins=50)
plt.title("Distribution of Article lengths")
plt.show()

print("Distribution of Summary-to-Article Compression Ratio")
seaborn.histplot(train_df['compression_ratio'], bins=200)
plt.xlim(0, 0.3)
plt.title("Distribution (Character-Level, Zoomed)")
plt.show()
