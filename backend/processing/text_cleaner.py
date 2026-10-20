import re


def clean_text(text):
    """
    Clean extracted document text while preserving meaningful content.
    """

    # Remove long underscore separator lines
    text = re.sub(r"_{5,}", "", text)

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove spaces at the beginning/end of each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines at the beginning/end
    text = "\n".join(lines).strip()

    return text