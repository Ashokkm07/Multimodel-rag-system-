
from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(
        self,
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize the multilingual embedding model.

        This model supports multiple languages and is
        suitable for multilingual RAG, including Tamil
        transcript + English query retrieval.
        """

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        print(
            "Embedding model loaded successfully."
        )

    def embed_text(self, text):
        """
        Generate an embedding for a single text.
        """

        return self.model.encode(
            text
        ).tolist()

    def embed_documents(self, texts):
        """
        Generate embeddings for multiple documents.
        """

        return self.model.encode(
            texts
        ).tolist()

