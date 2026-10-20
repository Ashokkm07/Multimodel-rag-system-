import pymupdf


def load_pdf(pdf_path):
    """
    Extract text from a PDF while preserving page numbers.
    """

    documents = []

    pdf = pymupdf.open(pdf_path)
    
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text().strip()

        if text:
            documents.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,
                    "page": page_number
                }
            })

    pdf.close()

    return documents