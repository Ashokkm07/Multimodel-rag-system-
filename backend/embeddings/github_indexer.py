from backend.embeddings.embedder import Embedder
from backend.vectorstore.github_chroma_store import (
    GitHubChromaStore
)


class GitHubIndexer:

    def __init__(self):

        self.embedder = Embedder()

        self.store = GitHubChromaStore()

    def index_repository(
        self,
        repository_url,
        chunks
    ):

        if not chunks:
            return {
                "status": "empty",
                "chunks": 0,
                "stored": 0
            }

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(
            "\nGenerating GitHub code embeddings..."
        )

        embeddings = (
            self.embedder.embed_documents(
                texts
            )
        )

        print(
            f"Embeddings generated: "
            f"{len(embeddings)}"
        )

        stored = self.store.add_chunks(
            repository_url=repository_url,
            chunks=chunks,
            embeddings=embeddings
        )

        return {
            "status": "completed",
            "chunks": len(chunks),
            "stored": stored
        }

    def search(
        self,
        repository_url,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedder.embed_text(
                query
            )
        )

        return self.store.search(
            query_embedding=query_embedding,
            repository_url=repository_url,
            top_k=top_k
        )

    def count(self):
        return self.store.count()