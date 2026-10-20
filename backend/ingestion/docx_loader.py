from docx import Document


def load_docx(docx_path):
    """
    Extract text from a DOCX document.
    """

    document = Document(docx_path)

    documents = []

    for paragraph_number, paragraph in enumerate(
        document.paragraphs,
        start=1
    ):
        text = paragraph.text.strip()

        if text:
            documents.append({
                "text": text,
                "metadata": {
                    "source": str(docx_path),
                    "paragraph": paragraph_number
                }
            })

    return documents