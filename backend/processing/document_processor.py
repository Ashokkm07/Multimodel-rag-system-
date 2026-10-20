from backend.processing.text_cleaner import clean_text
from backend.processing.chunker import chunk_documents


def process_documents(documents):
    """
    Clean extracted documents and split them into chunks.

    Pipeline:
        Extracted documents
                ↓
           Text cleaning
                ↓
             Chunking
    """

    cleaned_documents = []

    for document in documents:

        text = document.get("text", "").strip()

        if not text:
            continue

        cleaned_text = clean_text(text)

        if not cleaned_text:
            continue

        cleaned_documents.append({
            "text": cleaned_text,
            "metadata": document.get("metadata", {})
        })

    chunks = chunk_documents(cleaned_documents)

    return chunks