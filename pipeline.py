import time
import traceback

from asr_stream import load_asr_model, stream_transcripts
from text_chunker import segment_sentences, update_buffer, reset_buffer
from summarizer import load_summarizer_model, summarize

SUMMARY_EVERY_N_SEGMENTS = 2  


def main():
    print("[PIPELINE] Initializing components...")

    # Load ASR model 
    asr_model = load_asr_model()

    # 2) Load summarizer model 
    sum_tokenizer, sum_model = load_summarizer_model(
        "models/finalized_m2_reduced_lead3.sav"
    )

    # 3) Reset any existing sentence buffer
    reset_buffer()

    print("[PIPELINE] Starting real-time NoteBridge demo.")
    print("Press Ctrl+C to stop.\n")

    segment_counter = 0

    try:
        # Stream raw text from ASR
        for asr_text in stream_transcripts(asr_model):
            print(f"[ASR RAW] {asr_text}")

            # Phase 2: segment into sentences
            sentences = segment_sentences(asr_text)
            if not sentences:
                continue

            # Update rolling window
            rolling_context = update_buffer(sentences)
            print(f"[CONTEXT] {rolling_context}")

            segment_counter += 1

            if segment_counter % SUMMARY_EVERY_N_SEGMENTS != 0:
                continue

            summary = summarize(rolling_context, sum_tokenizer, sum_model)
            if summary.strip():
                print("\n================ ROLLING SUMMARY ================")
                print(summary)
                print("=================================================\n")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[PIPELINE] Stopped by user.")
    except Exception as e:
        print("[PIPELINE] Error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
