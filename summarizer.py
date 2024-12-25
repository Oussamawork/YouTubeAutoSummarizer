from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_transcript(transcript):
    """
    Summarizes a given transcript using the BART model.

    Args:
        transcript (str): The full transcript of the video.

    Returns:
        str: A summarized version of the transcript.
    """
    max_input_length = 1024  # Limit for BART model
    transcript_chunks = [transcript[i:i + max_input_length] for i in range(0, len(transcript), max_input_length)]
    summary = ""
    
    for chunk in transcript_chunks:
        summarized = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += summarized[0]['summary_text'] + " "
    
    return summary.strip()
