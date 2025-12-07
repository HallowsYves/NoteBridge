import time
import traceback
from datetime import datetime

from asr_stream import load_asr_model, stream_transcripts
from text_chunker import (
    segment_sentences,
    update_buffer,
    reset_buffer,
    clean_asr_text,
)
from summarizer import load_summarizer_model, summarize
import re

def clean_context_for_summary(text):
    """
    Remove fragments, stutters, incomplete thoughts,
    and keep only well-formed sentences.
    """
    sentences = re.split(r"[.!?]", text)
    cleaned = []

    for s in sentences:
        s = s.strip()

        # Skip tiny fragments
        if len(s.split()) < 4:
            continue

        # Skip stutter patterns
        if re.search(r"\b(\w+)\s+\1\b", s.lower()):
            continue

        # Skip nonsense fragments (ends mid-word)
        if re.search(r"[A-Za-z]-$", s):
            continue

        cleaned.append(s)

    return ". ".join(cleaned)




SUMMARY_EVERY_N_SEGMENTS = 12     # rolling summary frequency
SILENCE_THRESHOLD = 3.0          # seconds of no speech triggers FINAL SUMMARY


def pretty_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def save_final_summary(text):
    """Write the final summary to a text file."""
    filename = f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[SUMMARY] Final summary saved to: {filename}")


def main():
    pretty_header("NoteBridge Real-Time Summarization Pipeline")

    print(f"[{timestamp()}] Initializing ASR + Summarizer Models...")
    asr_model = load_asr_model()

    tokenizer, model = load_summarizer_model()
    print("[SUMMARIZER] Model ready.\n")

    reset_buffer()

    print(f"[{timestamp()}] System Ready. Listening...\n")
    print("Press Ctrl+C to stop.\n")

    segment_counter = 0
    last_speech_time = time.time()
    final_summary_done = False

    try:
        for asr_text in stream_transcripts(asr_model):

            # Track time of last detected speech
            last_speech_time = time.time()

            print(f"[{timestamp()}]  ASR: {asr_text}")

            # Clean text before segmentation
            cleaned = clean_asr_text(asr_text)
            if not cleaned:
                continue

            sentences = segment_sentences(cleaned)
            if not sentences:
                continue

            # Update rolling buffer
            context = update_buffer(sentences)
            print(f"[{timestamp()}]  Context Window:")
            print(f"   {context}\n")

            segment_counter += 1

            # Rolling summary every N segments
            if segment_counter % SUMMARY_EVERY_N_SEGMENTS == 0:

                cleaned_context = clean_context_for_summary(context)

                if len(cleaned_context.split()) < 6:
                    # Not enough meaningful content to summarize
                    continue

                summary = summarize(cleaned_context, tokenizer, model)

                pretty_header("ROLLING SUMMARY")
                print(summary)
                print("=" * 60 + "\n")


            if not final_summary_done and (time.time() - last_speech_time) > SILENCE_THRESHOLD:
                cleaned_context = clean_context_for_summary(context)
                final_summary = summarize(cleaned_context, tokenizer, model).strip()


                pretty_header("FINAL SUMMARY")
                print(final_summary)
                print("=" * 60 + "\n")

                save_final_summary(final_summary)
                final_summary_done = True
                break  

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[PIPELINE] Stopped by user.")

    except Exception:
        print("[PIPELINE] Error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
