import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use CPU for real-time safety
device = torch.device("cpu")

GEN_MAX_TOKENS = 48
NUM_BEAMS = 2


def load_summarizer_model(sav_path: str):
    """
    Loads tokenizer + model from your saved .sav metadata file.
    Returns: tokenizer, model
    """
    print(f"[SUMMARIZER] Loading model metadata: {sav_path}")

    with open(sav_path, "rb") as f:
        meta = pickle.load(f)

    model_dir = meta["dir"]
    print(f"[SUMMARIZER] Loading HF model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    print("[SUMMARIZER] Model ready.")
    return tokenizer, model



from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_summarizer_model(path=None):
    """
    We ignore path because we use flan-t5-small now.
    """
    print("[SUMMARIZER] Loading flan-t5-small...")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    print("[SUMMARIZER] Ready.")
    return tokenizer, model


def summarize(text, tokenizer, model, max_len=128):
    if not text or len(text.strip()) == 0:
        return ""

    input_text = "summarize: " + text.strip()

    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
        inputs,
        max_length=max_len,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

