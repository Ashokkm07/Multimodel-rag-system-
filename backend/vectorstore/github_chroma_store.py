import hashlib
from pathlib import Path

import chromadb


class GitHubChromaStore:

    def __init__(
        self,
        persist_directory="data/chroma"
    ):

        Path(
            persist_directory
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="github_code"
            )
        )

    def generate_chunk_id(
        self,
        repository_url,
        chunk
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        unique_string = "|".join([
            str(repository_url),
            str(metadata.get("file", "")),
            str(metadata.get("chunk_type", "")),
            str(metadata.get("name", "")),
            str(metadata.get("function", "")),
            str(metadata.get("class", "")),
            str(metadata.get("start_line", "")),
            str(metadata.get("end_line", "")),
            str(chunk.get("text", ""))
        ])

        hash_value = hashlib.sha256(
            unique_string.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"github_chunk_{hash_value}"
        )

    def add_chunks(
        self,
        repository_url,
        chunks,
        embeddings
    ):

        if len(chunks) != len(embeddings):

            raise ValueError(
                "Number of chunks and embeddings "
                "must be identical."
            )

        if not chunks:

            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            metadata = chunk.get(
                "metadata",
                {}
            )

            chunk_id = (
                self.generate_chunk_id(
                    repository_url,
                    chunk
                )
            )

            ids.append(chunk_id)

            documents.append(
                chunk["text"]
            )

            chroma_metadata = {
                "repository_url": str(
                    repository_url
                ),
                "file": str(
                    metadata.get(
                        "file",
                        ""
                    )
                ),
                "language": str(
                    metadata.get(
                        "language",
                        ""
                    )
                ),
                "chunk_type": str(
                    metadata.get(
                        "chunk_type",
                        ""
                    )
                ),
                "name": str(
                    metadata.get(
                        "name",
                        ""
                    )
                ),
                "function": str(
                    metadata.get(
                        "function",
                        ""
                    )
                ),
                "class": str(
                    metadata.get(
                        "class",
                        ""
                    )
                ),
                "start_line": str(
                    metadata.get(
                        "start_line",
                        ""
                    )
                ),
                "end_line": str(
                    metadata.get(
                        "end_line",
                        ""
                    )
                ),
                "chunk_id": str(
                    metadata.get(
                        "chunk_id",
                        ""
                    )
                )
            }

            metadatas.append(
                chroma_metadata
            )

        if len(ids) != len(set(ids)):

            raise ValueError(
                "Duplicate GitHub chunk IDs "
                "were generated."
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(ids)

    def count(self):

        return self.collection.count()

    def search(
        self,
        query_embedding,
        repository_url=None,
        top_k=5
    ):

        query_kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": top_k
        }

        if repository_url:

            query_kwargs["where"] = {
                "repository_url": str(
                    repository_url
                )
            }

        results = self.collection.query(
            **query_kwargs
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            retrieved.append({
                "text": document,
                "metadata": metadata,
                "distance": distance
            })

        return retrieved