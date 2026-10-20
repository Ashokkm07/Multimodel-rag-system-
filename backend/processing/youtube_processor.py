
from backend.processing.text_cleaner import clean_text


def process_youtube_transcript(documents):
    """
    Clean YouTube transcript segments while
    preserving their metadata.
    """

    processed_documents = []

    for document in documents:

        text = document.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        cleaned_text = clean_text(
            text
        )

        if not cleaned_text:
            continue

        metadata = document.get(
            "metadata",
            {}
        ).copy()

        processed_documents.append({
            "text": cleaned_text,
            "metadata": metadata
        })

    return processed_documents

