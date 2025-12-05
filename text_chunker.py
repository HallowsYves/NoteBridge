import nltk
import re

PARTIAL_BUFFER = ""
MIN_CHARS_FOR_SENTENCE = 40  

nltk.download("punkt")

ROLLING_WINDOW = 5

sentence_buffer = []


def is_duplicate_sentence(new_sent, buffer):
    """
    Prevents adding duplicate or nearly identical sentences to the rolling window.
    """
    new_clean = new_sent.lower().strip()
    for s in buffer:
        if new_clean == s.lower().strip():
            return True
        if new_clean in s.lower().strip() or s.lower().strip() in new_clean:
            return True
    return False

def segment_sentences(asr_text: str):
    """
    Accumulates ASR text into a temporary buffer.
    Only returns full, strong sentences when detected.
    This avoids fragments like 'which is the note' or 'and then'.
    """
    global PARTIAL_BUFFER

    if not asr_text or not asr_text.strip():
        return []

    PARTIAL_BUFFER += " " + asr_text.strip()

    if len(PARTIAL_BUFFER) < MIN_CHARS_FOR_SENTENCE:
        return []

    sentences = nltk.sent_tokenize(PARTIAL_BUFFER)

    if len(sentences) <= 1:
        return []

    complete = sentences[:-1]
    PARTIAL_BUFFER = sentences[-1]

    complete = [s for s in complete if len(s.split()) >= 3]

    return complete




def update_buffer(new_sentences):
    global sentence_buffer

    for sent in new_sentences:
        if len(sent.split()) < 3:
            continue

        if is_duplicate_sentence(sent, sentence_buffer):
            continue

        sentence_buffer.append(sent)

    if len(sentence_buffer) > ROLLING_WINDOW:
        sentence_buffer = sentence_buffer[-ROLLING_WINDOW:]

    return " ".join(sentence_buffer)



def reset_buffer():
    """Clears the rolling window buffer."""
    global sentence_buffer
    sentence_buffer = []
