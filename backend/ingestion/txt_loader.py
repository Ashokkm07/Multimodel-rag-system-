def load_txt(txt_path):
    """
    Extract text from a TXT file.
    """

    documents = []

    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read().strip()

    if text:
        documents.append({
            "text": text,
            "metadata": {
                "source": str(txt_path),
                "type": "txt"
            }
        })

    return documents