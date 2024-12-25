from transformers import pipeline
from log import log_info

def summarize_transcript(transcript):
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    max_input_length = 1024  # Limit for BART model
    transcript_chunks = [transcript[i:i + max_input_length] for i in range(0, len(transcript), max_input_length)]
    
    summary = ""
    for chunk in transcript_chunks:
        summarized = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += summarized[0]['summary_text'] + " "
    
    return summary.strip()

def second_pass_summarize(summary_text):
    # Load a text-to-text model pipeline (FLAN-T5 or T5, for example)
    bullet_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

    max_input_length = 1024
    summary_chunks = [summary_text[i : i + max_input_length] for i in range(0, len(summary_text), max_input_length)]
    
    combined_bullet_points = ""
    for i, chunk in enumerate(summary_chunks, 1):
        log_info(print(f"[INFO] Processing second-pass chunk {i}/{len(summary_chunks)}"))

        # Tweak max_length and min_length to control bullet point length
        result = bullet_pipeline(chunk, max_length=200, min_length=50, do_sample=False)

        bullet_points = result[0]['generated_text']
        combined_bullet_points += bullet_points.strip() + "\n\n"
    
    return combined_bullet_points.strip()