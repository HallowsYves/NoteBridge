from asr_stream import load_asr_model, stream_transcripts

model = load_asr_model()

for text in stream_transcripts(model):
    print("ASR:", text)
