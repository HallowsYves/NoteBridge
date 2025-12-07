import re
from collections import deque

MAX_SENTENCES = 6

_sentence_buffer = deque()


def reset_buffer():
    """Clear the global rolling buffer."""
    _sentence_buffer.clear()


def clean_asr_text(text):
    """
    Cleans noisy ASR fragments:
    - removes stutters
    - removes repeated words
    - filters fragments shorter than 3 chars
    - removes hyphenated partial words
    """
    if not text:
        return ""

    text = text.strip()

    # Remove trailing hyphen fragments like "chemical-"
    text = re.sub(r"\b\w+-\s*$", "", text)

    # Tokenize
    words = text.split()

    cleaned_words = []
    prev = None

    for w in words:
        # Remove single-letter or two-letter noise words
        if len(w) < 3:
            continue
        
        # Remove stutters ("Photos Photosynthesis")
        if w.lower() == prev:
            continue
        
        cleaned_words.append(w)
        prev = w.lower()

    return " ".join(cleaned_words).strip()



def segment_sentences(text: str):
    """
    Naive sentence segmentation based on punctuation.
    You can replace this with a better segmenter if needed,
    but this is sufficient for the project.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation
    parts = re.split(r"([.!?])", text)
    sentences = []
    current = ""

    for part in parts:
        if not part:
            continue
        current += part
        if part in ".!?":
            s = current.strip()
            if len(s) > 0:
                sentences.append(s)
            current = ""

    # Catch any trailing fragment
    tail = current.strip()
    if tail:
        sentences.append(tail)

    return sentences


def update_buffer(new_sentences):
    """
    Update the rolling buffer with new sentences and return
    the concatenated context string.
    """
    for s in new_sentences:
        if not s or len(s.strip()) == 0:
            continue
        _sentence_buffer.append(s.strip())
        while len(_sentence_buffer) > MAX_SENTENCES:
            _sentence_buffer.popleft()

    return " ".join(_sentence_buffer)
