import nltk
import re

# Download tokenizer if first time
nltk.download("punkt")

# Rolling window size (number of sentences kept)
ROLLING_WINDOW = 5

# Internal buffer for the sliding window
sentence_buffer = []


def segment_sentences(asr_text: str):
    """
    Takes raw ASR output and returns a list of clean sentences.
    Whisper often outputs fragments; NLTK helps correct that.
    """
    if not isinstance(asr_text, str) or not asr_text.strip():
        return []

    # Basic cleaning
    cleaned = asr_text.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Segment into sentences using NLTK
    sentences = nltk.sent_tokenize(cleaned)

    return sentences



def update_buffer(new_sentences):
    """
    Adds one or multiple sentences to the rolling buffer.
    Maintains the fixed-length window.
    Returns the concatenated text for summarization.
    """
    global sentence_buffer

    for sent in new_sentences:
        sentence_buffer.append(sent)

    # Keep only the last N sentences
    if len(sentence_buffer) > ROLLING_WINDOW:
        sentence_buffer = sentence_buffer[-ROLLING_WINDOW:]

    # Return window text for summarization
    return " ".join(sentence_buffer)



def reset_buffer():
    """Clears the rolling window buffer."""
    global sentence_buffer
    sentence_buffer = []
