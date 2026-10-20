from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


class DocumentReindexer:

    def __init__(self):
        """
        Initialize the embedding model and ChromaDB store.
        """

        self.embedder = Embedder()
        self.store = ChromaStore()

    def reindex_document(
        self,
        document_id,
        source_path
    ):
        """
        Repair and re-index an existing document.

        Only chunks belonging to the specified source path
        are updated.

        Existing:
        - chunk IDs
        - text
        - page numbers

        are preserved.

        Metadata is updated with:
        - document_id
        - source_type
        """

        document_id = str(
            document_id
        )

        print(
            f"\nRe-indexing document: "
            f"{document_id}"
        )

        print(
            f"Source path: "
            f"{source_path}"
        )

        # -------------------------------------------------
        # 1. Retrieve all existing ChromaDB records
        # -------------------------------------------------

        results = self.store.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = results.get(
            "documents",
            []
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        ids = results.get(
            "ids",
            []
        )

        print(
            f"Total chunks in ChromaDB: "
            f"{len(documents)}"
        )

        # -------------------------------------------------
        # 2. Find chunks belonging to this document
        # -------------------------------------------------

        selected_documents = []
        selected_metadatas = []
        selected_ids = []

        normalized_source = str(
            source_path
        ).replace(
            "/",
            "\\"
        ).lower()

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas
        ):

            source = str(
                metadata.get(
                    "source",
                    ""
                )
            ).replace(
                "/",
                "\\"
            ).lower()

            if source == normalized_source:

                selected_ids.append(
                    chunk_id
                )

                selected_documents.append(
                    document
                )

                selected_metadatas.append(
                    metadata
                )

        print(
            f"Document chunks found: "
            f"{len(selected_documents)}"
        )

        if not selected_documents:

            print(
                "\n❌ No chunks found for this document."
            )

            return {
                "status": "empty",
                "document_id": document_id,
                "chunks": 0,
                "updated": 0
            }

        # -------------------------------------------------
        # 3. Update metadata
        # -------------------------------------------------

        updated_metadatas = []

        for metadata in selected_metadatas:

            updated_metadata = metadata.copy()

            updated_metadata["document_id"] = (
                document_id
            )

            updated_metadata["source_type"] = (
                "document"
            )

            updated_metadatas.append(
                updated_metadata
            )

        print(
            "Document metadata updated."
        )

        # -------------------------------------------------
        # 4. Generate embeddings
        # -------------------------------------------------

        print(
            "\nGenerating embeddings..."
        )

        embeddings = self.embedder.embed_documents(
            selected_documents
        )

        print(
            f"Embeddings generated: "
            f"{len(embeddings)}"
        )

        # -------------------------------------------------
        # 5. Update only these chunks
        # -------------------------------------------------

        print(
            "\nUpdating ChromaDB..."
        )

        self.store.collection.upsert(
            ids=selected_ids,
            documents=selected_documents,
            embeddings=embeddings,
            metadatas=updated_metadatas
        )

        print(
            f"ChromaDB updated: "
            f"{len(selected_ids)} chunks"
        )

        return {
            "status": "completed",
            "document_id": document_id,
            "chunks": len(selected_documents),
            "updated": len(selected_ids)
        }