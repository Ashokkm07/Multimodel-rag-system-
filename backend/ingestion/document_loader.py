from pathlib import Path

from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.docx_loader import load_docx
from backend.ingestion.pptx_loader import load_pptx
from backend.ingestion.txt_loader import load_txt
from backend.ingestion.csv_loader import load_csv


def load_document(file_path):
    """
    Automatically select the correct loader based on file extension.
    """

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    elif extension == ".pptx":
        return load_pptx(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    elif extension in [".csv", ".tsv"]:
        return load_csv(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )