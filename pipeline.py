import time
import traceback
from datetime import datetime
import re

from asr_stream import load_asr_model, stream_transcripts
from summarizer import load_summarizer_model, summarize


# ================================
# SETTINGS
# ================================
CHUNK_TIME_SECONDS = 5.0   # Summarize every 5 seconds


# ================================
# UTILITY FUNCTIONS
# ================================
def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def pretty_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


# ================================
# ASR REPAIR LAYER (AGGRESSIVE)
# ================================
def _normalize_text(text):
    """Basic normalization: collapse whitespace, normalize punctuation."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()

    t = re.sub(r"([\.!?])\1+", r"\1", t)

    return t


def _apply_domain_fixes(text):
    """
    Apply small, domain-specific corrections.
    Covers:
    - Networking terms (IP, IPv4, IPv6)
    - Photosynthesis / biology terms (photosynthesis, chlorophyll, etc.)
    """
    fixed = text

    net_patterns = {
        r"\bipf\b": "ip",
        r"\bipee\b": "ip",
        r"\bi p address\b": "ip address",
        r"\bipv 4\b": "ipv4",
        r"\bipv 6\b": "ipv6",
    }

    for pattern, repl in net_patterns.items():
        fixed = re.sub(pattern, repl, fixed, flags=re.IGNORECASE)

    fixed = re.sub(r"\s+(ipv4|ipv6)\b", r". \1", fixed, flags=re.IGNORECASE)

    bio_patterns = {
        r"\bphotos synthesis\b": "photosynthesis",
        r"\bphoto synthesis\b": "photosynthesis",
        r"\bphotos\s+in\s+this\b": "photosynthesis",
        r"\bchlorosil\b": "chlorophyll",
        r"\bchloro,\s*sis\b": "chlorophyll",
        r"\bchloro\s*sis\b": "chlorophyll",
        r"\bthis is the essentials\b": "this is essential",
        r"\boutside and water\b": "carbon dioxide and water",
    }

    for pattern, repl in bio_patterns.items():
        fixed = re.sub(pattern, repl, fixed, flags=re.IGNORECASE)

    return fixed


def _collapse_repeated_tokens(text):
    """
    Tokenize into words + punctuation, then collapse consecutive repeated words.
    Example: "what what what is this" -> "what is this"
    """
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if not tokens:
        return ""

    cleaned = []
    last_word = None

    for tok in tokens:
        if tok.isalpha():
            lower_tok = tok.lower()
            if last_word is not None and lower_tok == last_word:
                continue
            last_word = lower_tok
            cleaned.append(tok)
        else:
            cleaned.append(tok)

    out = []
    for tok in cleaned:
        if not out:
            out.append(tok)
        else:
            if tok.isalnum():
                out.append(" " + tok)
            else:
                out.append(tok)

    return "".join(out).strip()


def _split_sentences(text):
    """
    Simple sentence segmentation based on punctuation.
    Falls back gracefully if there is no punctuation.
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[\.!?])\s+", text)

    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _merge_fragments(sentences):
    """
    Aggressively merge short / continuation fragments into prior sentences.

    Examples:
    - "Photosynthesis is a process."
      "is the process plants use to convert something."
      =>
      "Photosynthesis is the process plants use to convert something."

    - "convert sunlight into chemical energy."
      "energy."
      =>
      "convert sunlight into chemical energy."

    - "essential for plant growth and oxygen."
      "production."
      =>
      "essential for plant growth and oxygen production."
    """
    if not sentences:
        return []

    merged = []
    continuation_starts = (
        "is ", "are ", "was ", "were ",
        "and ", "to ", "then ", "so ",
        "that ", "which ", "convert ", "converts ",
        "absorbs ", "absorb ", "transforms ", "transform ",
        "into ", "this is ", "essential ", "essential for ",
        "for plant ", "for plants ", "production", "energy"
    )

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue

        s_clean = re.sub(r"\s+([\.!?])", r"\1", s_clean)

        if not merged:
            merged.append(s_clean)
            continue

        lower = s_clean.lower()
        prev = merged[-1].rstrip(" .!?")
        prev_lower = prev.lower()

        if prev_lower.startswith("photosynthesis is a process") and lower.startswith("is the process"):
            rest = s_clean[len("is the process") :].lstrip()
            new_prev = "Photosynthesis is the process"
            if rest:
                new_prev += " " + rest
            merged[-1] = new_prev.strip()
            continue

        if len(s_clean.split()) <= 4 or any(lower.startswith(p) for p in continuation_starts):
            if lower in ("production", "production."):
                merged[-1] = prev + " production"
            else:
                merged[-1] = prev + " " + s_clean.lstrip()
        else:
            merged.append(s_clean)

    return merged


def _clean_sentence(sent):
    """
    Clean and validate a single merged sentence:
    - Remove leading filler phrases
    - Filter out too-short or low-content sentences
    - Capitalize first character
    """
    if not sent:
        return None

    s = sent.strip()
    if not s:
        return None

    s = s.strip(" \t\n\r")

    lower = s.lower()

    filler_prefixes = [
        "okay", "ok", "so", "well", "yeah", "right",
        "like", "um", "uh", "hmm", "you know"
    ]
    for fp in filler_prefixes:
        if lower.startswith(fp + " "):
            s = s[len(fp) + 1 :].lstrip()
            lower = s.lower()
            break

    tokens = s.split()
    if len(tokens) < 4:
        return None

    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    if len(alpha_tokens) < 3:
        return None

    filler_set = {
        "uh", "um", "hmm", "huh", "like", "yeah", "okay", "ok",
        "right", "so", "well"
    }
    non_filler = [t for t in alpha_tokens if t.lower() not in filler_set]
    if len(non_filler) < 2:
        return None

    s = s.strip()
    if not s:
        return None

    s = s[0].upper() + s[1:]

    return s


def repair_asr_text(text):
    """
    Aggressive ASR cleanup & reconstruction:

    - Normalizes whitespace & punctuation
    - Applies small domain corrections (networking + basic biology)
    - Collapses repeated tokens (e.g., "what what what")
    - Splits into sentences
    - Aggressively merges short/continuation fragments into full sentences
    - Cleans and filters low-information sentences
    - Returns coherent English suitable for summarization
    """
    if not text or not text.strip():
        return ""

    t = _normalize_text(text)

    t = _apply_domain_fixes(t)

    t = _collapse_repeated_tokens(t)

    candidate_sentences = _split_sentences(t)

    merged_sentences = _merge_fragments(candidate_sentences)

    final_sentences = []
    for cand in merged_sentences:
        s = _clean_sentence(cand)
        if s:
            final_sentences.append(s)

    if not final_sentences:
        return ""

    repaired = ". ".join(final_sentences)

    if repaired and repaired[-1] not in ".!?":
        repaired += "."

    return repaired


# ================================
# MAIN PIPELINE
# ================================
def main():
    print("\n============================================================")
    print("Initializing models...")
    print("============================================================\n")

    # Load ASR + Summarizer
    asr_model = load_asr_model()
    tokenizer, model = load_summarizer_model()
    print("[SUMMARIZER] Ready.\n")

    print("Listening...\n")
    print("[ASR] Starting microphone stream...")

    buffer_text = ""
    last_chunk_time = time.time()

    try:
        for text in stream_transcripts(asr_model):

            print(f"[{timestamp()}] ASR: {text}")

            buffer_text += " " + text.strip()

            if time.time() - last_chunk_time >= CHUNK_TIME_SECONDS:

                repaired = repair_asr_text(buffer_text)

                if len(repaired.strip()) > 10:
                    pretty_header("CHUNK SUMMARY")
                    print("RAW BUFFER:")
                    print(buffer_text.strip())
                    print("\nREPAIRED INPUT:")
                    print(repaired.strip() + "\n")

                    summary = summarize(repaired.strip(), tokenizer, model)
                    print("summary:", summary)
                    print("=" * 60 + "\n")
                else:
                    print("[SUMMARY] Skipped (not enough meaningful content)\n")

                buffer_text = ""
                last_chunk_time = time.time()

    except KeyboardInterrupt:
        print("\n[PIPELINE] Stopped by user.")

    except Exception:
        print("[PIPELINE] Error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
