from transformers import pipeline
from log import log_info, log_warn, log_error
from langdetect import detect


def translate_to_english(text):
    """
    Detects the language of the text and translates it to English using a pre-trained model.
    """
    try:
        detected_language = detect(text)
        if detected_language != "en":
            log_info(f"Detected language '{detected_language}'. Translating to English...")
            translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{detected_language}-en")
            translated = translator(text, max_length=1024)
            return translated[0]['translation_text']
        return text
    except Exception as e:
        log_error(f"Translation failed: {str(e)}. Returning original text.")
        return text


def sliding_window_chunking(text, window_size, overlap):
    """
    Splits text into overlapping chunks to preserve context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + window_size
        chunks.append(text[start:end])
        start += window_size - overlap  # Move the window forward with overlap
    return chunks


def summarize_transcript(transcript):
    """
    Summarizes a transcript using the BART model and ensures the output is in English.
    """
    log_info("Initializing the BART summarization pipeline...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    window_size = 1024
    overlap = 256
    log_info(f"Splitting transcript into chunks with window size {window_size} and overlap {overlap}...")
    transcript_chunks = sliding_window_chunking(transcript, window_size, overlap)

    summary = ""
    for idx, chunk in enumerate(transcript_chunks):
        try:
            log_info(f"Processing chunk {idx + 1}/{len(transcript_chunks)}...")
            summarized = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            chunk_summary = summarized[0]['summary_text']

            # Ensure the chunk summary is in English
            chunk_summary_english = translate_to_english(chunk_summary)
            summary += chunk_summary_english + " "
            log_info(f"Chunk {idx + 1} summarized successfully.")
        except Exception as e:
            log_warn(f"Failed to summarize chunk {idx + 1}: {str(e)}")

    log_info("All chunks processed. Finalizing the summary...")
    return summary.strip()


def second_pass_summarize(summary_text):
    """
    Extracts key ideas and formats them as concise bullet points using sliding window chunking.
    """
    log_info("Initializing BART for bullet-point summarization with sliding window...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    window_size = 1024
    overlap = 256
    log_info(f"Splitting summary text into chunks with window size {window_size} and overlap {overlap}...")
    summary_chunks = sliding_window_chunking(summary_text, window_size, overlap)

    combined_bullet_points = ""
    for i, chunk in enumerate(summary_chunks):
        try:
            log_info(f"Processing chunk {i + 1}/{len(summary_chunks)} for bullet points...")
            summarized = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            bullet_points = summarized[0]['summary_text']

            # Add the generated bullet points
            combined_bullet_points += bullet_points.strip() + "\n\n"
            log_info(f"Chunk {i + 1} processed successfully.")
        except Exception as e:
            log_warn(f"Failed to process chunk {i + 1}: {str(e)}")

    log_info("Bullet-point extraction completed.")
    return combined_bullet_points.strip()


if __name__ == "__main__":
    summary_text = ""
    
    log_info("Starting first-pass summarization...")
    summary = summarize_transcript(summary_text)
    log_info("Starting second-pass summarization to generate bullet points...")
    bullet_summary = second_pass_summarize(summary)
    log_info("Bullet-point summary generated:")
    print(bullet_summary)
