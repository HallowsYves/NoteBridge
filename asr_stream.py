import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Receives microphone audio in real time."""
    if status:
        print("[Audio Warning]", status)
    audio_queue.put(indata.copy())


def load_asr_model():
    print("[ASR] Loading Whisper model...")
    model = WhisperModel("medium.en", device="cpu", compute_type="float32")

    print("[ASR] Whisper ready.")
    return model


def stream_transcripts(model):
    """
    Captures audio from the microphone and streams partial transcripts.
    Yields one text segment at a time.
    """
    print("[ASR] Starting microphone stream...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    ):
        buffer = np.zeros((0, 1), dtype=np.float32)

        while True:
            # Pull new audio from queue
            chunk = audio_queue.get()
            buffer = np.concatenate([buffer, chunk])

            # Only process when buffer reaches 1-second chunks
            if len(buffer) >= CHUNK_SAMPLES:
                audio_to_process = buffer[:CHUNK_SAMPLES]
                buffer = buffer[CHUNK_SAMPLES:]

                # Flatten for Whisper input
                audio_flat = audio_to_process.flatten()

                # Run Whisper on this 1-second chunk
                segments, _ = model.transcribe(
                    audio_flat,
                    beam_size=1,
                    vad_filter=True  # helps trim silence
                )

                # Yield any recognized text
                for seg in segments:
                    if seg.text.strip():
                        yield seg.text.strip()
