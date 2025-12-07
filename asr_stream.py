import os
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HARDLINKS"] = "0"

SAMPLE_RATE = 16000

CHUNK_DURATION = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

OVERLAP_RATIO = 0.25
OVERLAP_SAMPLES = int(CHUNK_SAMPLES * OVERLAP_RATIO)

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Microphone callback: push raw audio chunks into queue."""
    if status:
        print("[Audio Warning]", status)
    audio_queue.put(indata.copy())


def is_silence(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Simple RMS-based VAD.
    Skip chunks that are very low energy (likely silence / noise).
    """
    if audio.size == 0:
        return True
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold


def is_informative(text: str) -> bool:
    """
    Filter out tiny or clearly garbage segments before they hit the buffer.
    - Very short fragments
    - Mostly non-alphabetic
    """
    t = text.strip()
    if len(t) < 4:
        return False

    alpha_count = sum(c.isalpha() for c in t)
    if alpha_count < 2:
        return False

    return True


def load_asr_model():
    print("[ASR] Loading Whisper model (optimized for accuracy)...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    print("[ASR] Whisper ready.")
    return model


def stream_transcripts(model):
    """
    Streaming ASR with:
    - Fixed-size chunks (1.2s)
    - 50% overlap between chunks
    - RMS-based silence gating
    - Filtering of non-informative text
    Yields cleaned text segments suitable for downstream summarization.
    """
    print("[ASR] Starting microphone stream...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=0,  
        callback=audio_callback,
    ):

        buffer = np.zeros((0, 1), dtype=np.float32)

        while True:
            try:
                chunk = audio_queue.get_nowait()
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, chunk])

            if len(buffer) < CHUNK_SAMPLES:
                continue

            audio_to_process = buffer[:CHUNK_SAMPLES]

            buffer = buffer[CHUNK_SAMPLES - OVERLAP_SAMPLES :]

            audio_flat = audio_to_process.flatten()

            if is_silence(audio_flat):
                continue

            segments, _ = model.transcribe(
                audio_flat,
                beam_size=1,
                vad_filter=True,
                temperature=0.0,                    
                compression_ratio_threshold=2.4      
            )


            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue

                if not is_informative(text):
                    continue

                yield text
