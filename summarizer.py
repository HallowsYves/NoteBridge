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



def summarize(text: str, tokenizer, model):
    """
    Generates a summary for the given text using your loaded model.
    Used for rolling, real-time summarization.
    """

    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Light cleaning: remove repeated periods, stutters, ASR artifacts
    cleaned = text
    cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace("...", ".")
    cleaned = cleaned.replace(" .", ".")
    cleaned = cleaned.strip()

    if len(cleaned.split()) < 6:
        return ""  

    # Prepare model input
    prompt = "summarize: " + cleaned

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256 
    ).to(device)

    # Generation (fast settings)
    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=GEN_MAX_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=True
        )

    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)
