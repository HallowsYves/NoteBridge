import os
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import evaluate
import nltk

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
nltk.download('punkt')

device = torch.device("cpu")
MAX_SOURCE_LEN_FULL = 256
MAX_SOURCE_LEN_REDUCED = 192
MAX_TARGET_LEN = 64

SEED = 42
EVAL_SIZE = 150   # <- size of test subset for evaluation (you can set to 100 if needed)


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def load_data(eval_size=EVAL_SIZE):
    data_files = {"train": "train_clean.parquet", "test": "test_clean.parquet"}
    dataset = load_dataset("HallowsYves/CPSC483-data", data_files=data_files)

    test_df = dataset["test"].to_pandas()

    # Sample a subset for evaluation to keep runtime manageable
    if eval_size is not None and len(test_df) > eval_size:
        test_df = test_df.sample(n=eval_size, random_state=SEED).reset_index(drop=True)

    print(f"[INFO] Using {len(test_df)} examples for evaluation.")
    return test_df


# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------
def load_pickled_model(sav_path):
    with open(sav_path, "rb") as f:
        meta = pickle.load(f)
    
    model_dir = meta["dir"]
    print(f"[INFO] Loading model from: {model_dir} | note={meta.get('note')}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    return tokenizer, model, meta


# ------------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------------
def generate_summaries(tokenizer, model, texts, max_source_len, batch_size=4):
    preds = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [
            "summarize: " + (t if isinstance(t, str) else "")
            for t in batch_texts
        ]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_source_len,
            padding=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=48,   # shorter summaries for speed
                num_beams=2,         # cheaper decoding
                early_stopping=True
            )

        for seq in gen_ids:
            preds.append(tokenizer.decode(seq, skip_special_tokens=True))

    return preds


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------
rouge = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

def compute_metrics(preds, refs, model_name):
    rouge_res = rouge.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True
    )

    bleu_res = bleu_metric.compute(
        predictions=preds,               
        references=[[r] for r in refs]   
    )

    gen_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]

    return {
        "model": model_name,
        "rouge1": rouge_res["rouge1"],
        "rouge2": rouge_res["rouge2"],
        "rougeL": rouge_res["rougeL"],
        "bleu": bleu_res["bleu"],
        "avg_gen_len": np.mean(gen_lens),
        "avg_ref_len": np.mean(ref_lens),
        "compression": np.mean(gen_lens) / np.mean(ref_lens),
    }



# ------------------------------------------------------------------
# VISUALIZATIONS
# ------------------------------------------------------------------
def build_visualizations(metrics_df, rougeL_full, rougeL_red, article_lens):
    # --- Visualization 1: Bar Chart ---
    metric_cols = ["rouge1", "rouge2", "rougeL", "bleu"]
    ax = metrics_df.set_index("model")[metric_cols].plot(kind="bar", figsize=(8, 5))
    ax.set_title("Summarization Metrics Comparison")
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig("viz_bar_metrics.png")
    plt.close()

    # --- Visualization 2: Boxplot (ROUGE-L) ---
    plt.figure(figsize=(6, 5))
    plt.boxplot([rougeL_full, rougeL_red], labels=["M1_full", "M2_lead3"])
    plt.title("ROUGE-L Distribution")
    plt.ylabel("ROUGE-L")
    plt.tight_layout()
    plt.savefig("viz_box_rougel.png")
    plt.close()

    # --- Visualization 3: Scatter Plot ---
    plt.figure(figsize=(7, 5))
    plt.scatter(article_lens, rougeL_full, alpha=0.5, label="M1_full")
    plt.scatter(article_lens, rougeL_red, alpha=0.5, marker="x", label="M2_lead3")
    plt.xlabel("Article Length (words)")
    plt.ylabel("ROUGE-L")
    plt.title("ROUGE-L vs Article Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig("viz_scatter_rouge_vs_len.png")
    plt.close()

    print("[INFO] Saved 3 visualization files.")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("[1] Loading data...")
    test_df = load_data()

    references = test_df["highlights"].tolist()
    articles = test_df["article"].tolist()

    print("[2] Loading models...")
    tok_full, mdl_full, _ = load_pickled_model("models/finalized_m1_full_distilbart.sav")
    tok_red,  mdl_red,  _ = load_pickled_model("models/finalized_m2_reduced_lead3.sav")

    print("[3] Running inference...")
    preds_full = generate_summaries(tok_full, mdl_full, articles, MAX_SOURCE_LEN_FULL, batch_size=4)
    preds_red  = generate_summaries(tok_red,  mdl_red,  articles, MAX_SOURCE_LEN_REDUCED, batch_size=4)

    print("[4] Computing global metrics...")
    m_full = compute_metrics(preds_full, references, "M1_full")
    m_red  = compute_metrics(preds_red,  references, "M2_lead3")

    metrics_df = pd.DataFrame([m_full, m_red])
    metrics_df.to_csv("metrics_results.csv", index=False)
    print(metrics_df)

    # Per-example ROUGE-L (optionally on a subset for speed)
    subset_n = min(80, len(articles))   # smaller subset for per-example stats
    subset_refs  = references[:subset_n]
    subset_full  = preds_full[:subset_n]
    subset_red   = preds_red[:subset_n]
    subset_lens  = [len(a.split()) for a in articles[:subset_n]]

    print(f"[4b] Computing per-example ROUGE-L on {subset_n} examples...")
    rouge_full_per = rouge.compute(
        predictions=subset_full,
        references=subset_refs,
        use_stemmer=True,
        use_aggregator=False
    )["rougeL"]

    rouge_red_per = rouge.compute(
        predictions=subset_red,
        references=subset_refs,
        use_stemmer=True,
        use_aggregator=False
    )["rougeL"]

    print("[5] Creating visualizations...")
    build_visualizations(metrics_df, rouge_full_per, rouge_red_per, subset_lens)

    print("[DONE] Evaluation complete. Files saved:")
    print(" - metrics_results.csv")
    print(" - viz_bar_metrics.png")
    print(" - viz_box_rougel.png")
    print(" - viz_scatter_rouge_vs_len.png")


if __name__ == "__main__":
    main()
