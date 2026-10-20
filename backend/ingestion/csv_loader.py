import pandas as pd


def load_csv(csv_path, batch_size=1000):
    """
    Load a large CSV/TSV dataset in batches.

    Each row becomes one document with its original
    columns preserved as metadata.
    """

    documents = []

    for dataframe in pd.read_csv(
        csv_path,
        sep=None,
        engine="python",
        chunksize=batch_size,
        on_bad_lines="warn"
    ):

        # Remove completely empty rows
        dataframe = dataframe.dropna(how="all")

        # Remove completely empty columns
        dataframe = dataframe.dropna(axis=1, how="all")

        for _, row in dataframe.iterrows():

            sentence = row.get("SENTENCE")

            if pd.isna(sentence):
                continue

            sentence = str(sentence).strip()

            if not sentence:
                continue

            metadata = {
                "source": str(csv_path)
            }

            # Preserve every useful column
            for column in dataframe.columns:
                value = row[column]

                if pd.notna(value):
                    metadata[column.lower()] = str(value)

            documents.append({
                "text": sentence,
                "metadata": metadata
            })

    return documents