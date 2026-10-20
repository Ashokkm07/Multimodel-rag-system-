
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ImageChunker:
    """
    Splits combined OCR + vision knowledge into retrieval-friendly chunks.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(self, content_result):
        if not content_result:
            raise ValueError("Image content result cannot be empty.")

        text = content_result.get("text", "").strip()

        if not text:
            return []

        base_metadata = dict(
            content_result.get("metadata", {})
        )

        chunks = self.splitter.split_text(text)

        result = []

        for index, chunk_text in enumerate(chunks):
            metadata = {
                **base_metadata,
                "source_type": "image",
                "chunk_index": index,
                "chunk_count": len(chunks),
            }

            result.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

        return result
