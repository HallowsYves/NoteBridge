import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cpu")

def load_summarizer_model(path=None):
    """
    Load a robust summarizer (flan-t5-base).
    The path is ignored because we are no longer using the .sav model.
    """
    print("[SUMMARIZER] Loading flan-t5-base...")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    model.eval()

    print("[SUMMARIZER] Ready.")
    return tokenizer, model


def summarize(text, tokenizer, model, max_len=128):
    if not text or len(text.strip()) == 0:
        return ""

    input_text = "summarize: " + text.strip()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
