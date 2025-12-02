from asr_stream import load_asr_model, stream_transcripts
from text_chunker import segment_sentences, update_buffer
import nltk


nltk.download("punkt")
nltk.download("punkt_tab")


model = load_asr_model()

for text in stream_transcripts(model):
    sents = segment_sentences(text)
    chunk_for_summary = update_buffer(sents)
    print("\n[ROLLING INPUT]:", chunk_for_summary)
