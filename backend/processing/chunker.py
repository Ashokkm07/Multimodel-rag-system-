from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    """
    Split cleaned documents into smaller chunks while preserving metadata.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        split_texts = text_splitter.split_text(text)

        for chunk_number, chunk_text in enumerate(split_texts, start=1):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk": chunk_number
                }
            })

    return chunks