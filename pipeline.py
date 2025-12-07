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


# ============================================================
# NEW STRICT SPAM FILTER (Option A)
# ============================================================
def _is_spam_fragment(text: str) -> bool:
    """
    Returns True if the fragment is clearly ASR hallucination spam.
    Spam examples:
        - repeated phrases ("glory to you" 10+ times)
        - repeated tokens ("8-bit" repeated many times)
        - same word repeated 3+ times in the same fragment
        - extremely long repetition chains
    """
    if not text:
        return True

    t = text.lower().strip()

    # 1. Detect repeated tokens (e.g. "8-bit 8-bit 8-bit...")
    tokens = t.split()
    for tok in set(tokens):
        if tokens.count(tok) >= 3:
            return True

    # 2. Detect long repeated phrase blocks ("glory to you" etc.)
    if len(tokens) > 20:
        # More than 20 tokens AND highly repetitive = spam
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.5:
            return True

    # 3. Detect non-language artifacts such as numeric or hyphen spam
    if re.search(r"\b\d+-bit\b", t):
        return True

    if re.search(r"([a-z]+ ){5,}\1", t):
        return True

    # 4. If text has < 3 alphabetic words → garbage
    alpha_words = [tok for tok in tokens if any(c.isalpha() for c in tok)]
    if len(alpha_words) < 3:
        return True

    return False


# ================================
# ASR REPAIR LAYER (AGGRESSIVE)
# ================================
def _normalize_text(text):
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"([\.!?])\1+", r"\1", t)
    return t


def _apply_domain_fixes(text):
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
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if not tokens:
        return ""

    cleaned = []
    last_word = None
    for tok in tokens:
        if tok.isalpha():
            low = tok.lower()
            if low == last_word:
                continue
            last_word = low
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
    if not text:
        return []
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _merge_fragments(sentences):
    if not sentences:
        return []

    merged = []
    continuation_starts = (
        "is ", "are ", "was ", "were ",
        "and ", "to ", "then ", "so ",
        "that ", "which ", "convert ",
        "absorbs ", "absorb ",
        "transforms ", "transform ",
        "into ", "this is ",
        "essential ", "essential for ",
        "for plant ", "for plants ",
        "production", "energy"
    )

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue

        # Skip entire fragment if spam
        if _is_spam_fragment(s_clean):
            continue

        s_clean = re.sub(r"\s+([\.!?])", r"\1", s_clean)

        if not merged:
            merged.append(s_clean)
            continue

        prev = merged[-1].rstrip(" .!?")
        prev_lower = prev.lower()
        low = s_clean.lower()

        # Merge short continuation
        if len(s_clean.split()) <= 4 or any(low.startswith(p) for p in continuation_starts):
            merged[-1] = prev + " " + s_clean.lstrip()
        else:
            merged.append(s_clean)

    return merged


def _clean_sentence(sent):
    if not sent:
        return None

    s = sent.strip().strip(" \t\n\r")
    lower = s.lower()

    filler_prefixes = [
        "okay", "ok", "so", "well", "yeah", "right",
        "like", "um", "uh", "hmm", "you know"
    ]
    for fp in filler_prefixes:
        if lower.startswith(fp + " "):
            s = s[len(fp) + 1:].lstrip()
            lower = s.lower()
            break

    tokens = s.split()
    if len(tokens) < 4:
        return None

    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    if len(alpha_tokens) < 3:
        return None

    filler_set = {"uh", "um", "hmm", "huh", "like", "yeah", "okay", "ok", "right", "so", "well"}
    if len([t for t in alpha_tokens if t.lower() not in filler_set]) < 2:
        return None

    return s[0].upper() + s[1:]


def repair_asr_text(text):
    if not text or not text.strip():
        return ""

    t = _normalize_text(text)
    t = _apply_domain_fixes(t)
    t = _collapse_repeated_tokens(t)

    fragmented = _split_sentences(t)
    merged = _merge_fragments(fragmented)

    final_sentences = []
    for sent in merged:
        clean = _clean_sentence(sent)
        if clean:
            final_sentences.append(clean)

    if not final_sentences:
        return ""

    out = ". ".join(final_sentences)
    if out[-1] not in ".!?":
        out += "."

    return out


# ================================
# MAIN PIPELINE
# ================================
def main():
    print("\n============================================================")
    print("Initializing models...")
    print("============================================================\n")

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
