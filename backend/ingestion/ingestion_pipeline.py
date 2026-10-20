from pathlib import Path

from backend.ingestion.document_loader import load_document
from backend.processing.document_processor import process_documents
from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore
from backend.utils.file_hash import calculate_file_hash
from backend.utils.database import Database


class IngestionPipeline:

    def __init__(self):
        """
        Initialize all components required for ingestion.
        """

        self.embedder = Embedder()
        self.store = ChromaStore()
        self.database = Database()

    def ingest(self, file_path):
        """
        Complete document ingestion pipeline.

        File
          ↓
        Hash
          ↓
        Duplicate Check
          ↓
        Load
          ↓
        Clean + Chunk
          ↓
        Add document metadata
          ↓
        Embeddings
          ↓
        ChromaDB
          ↓
        SQLite update
        """

        file_path = Path(file_path)

        print(f"\nLoading file: {file_path}")

        # -------------------------------------------------
        # 1. Calculate file hash
        # -------------------------------------------------

        file_hash = calculate_file_hash(file_path)

        print(f"File hash: {file_hash}")

        # -------------------------------------------------
        # 2. Check for duplicate
        # -------------------------------------------------

        duplicate = self.database.is_duplicate(file_hash)

        if duplicate["exists"]:

            print("\nDuplicate file detected.")

            print(
                f"Existing document: "
                f"{duplicate['filename']}"
            )

            print(
                f"Document ID: "
                f"{duplicate['id']}"
            )

            print(
                f"Status: "
                f"{duplicate['processing_status']}"
            )

            return {
                "status": "duplicate",
                "document_id": duplicate["id"],
                "filename": duplicate["filename"],
                "documents": 0,
                "chunks": 0,
                "stored": 0
            }

        print("\nNew file detected.")

        # -------------------------------------------------
        # 3. Register document as processing
        # -------------------------------------------------

        file_type = (
            file_path.suffix
            .lower()
            .replace(".", "")
        )

        document_id = self.database.add_document(
            filename=file_path.name,
            file_path=str(file_path),
            file_hash=file_hash,
            file_type=file_type,
            processing_status="processing",
            chunk_count=0
        )

        print(
            f"Document registered. ID: "
            f"{document_id}"
        )

        try:

            # -------------------------------------------------
            # 4. Load document
            # -------------------------------------------------

            documents = load_document(
                file_path
            )

            print(
                f"Documents extracted: "
                f"{len(documents)}"
            )

            if not documents:

                return {
                    "status": "empty",
                    "document_id": document_id,
                    "filename": file_path.name,
                    "documents": 0,
                    "chunks": 0,
                    "stored": 0
                }

            # -------------------------------------------------
            # 5. Clean and chunk
            # -------------------------------------------------

            chunks = process_documents(
                documents
            )

            print(
                f"Chunks created: "
                f"{len(chunks)}"
            )

            if not chunks:

                return {
                    "status": "empty",
                    "document_id": document_id,
                    "filename": file_path.name,
                    "documents": len(documents),
                    "chunks": 0,
                    "stored": 0
                }

            # -------------------------------------------------
            # 6. Add document-level metadata
            # -------------------------------------------------

            for chunk in chunks:

                chunk["metadata"]["document_id"] = str(
                    document_id
                )

                chunk["metadata"]["source_type"] = (
                    "document"
                )

            print(
                "Document metadata added to chunks."
            )

            # -------------------------------------------------
            # 7. Generate embeddings
            # -------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            print(
                "Generating embeddings..."
            )

            embeddings = self.embedder.embed_documents(
                texts
            )

            print(
                f"Embeddings generated: "
                f"{len(embeddings)}"
            )

            # -------------------------------------------------
            # 8. Store in ChromaDB
            # -------------------------------------------------

            print(
                "Storing in ChromaDB..."
            )

            stored = self.store.add_chunks(
                chunks,
                embeddings
            )

            print(
                f"Chunks stored: "
                f"{stored}"
            )

            # -------------------------------------------------
            # 9. Update SQLite record
            # -------------------------------------------------

            self.database.update_processing_status(
                document_id=document_id,
                processing_status="completed",
                chunk_count=stored
            )

            print(
                "Database status updated: completed"
            )

            # -------------------------------------------------
            # 10. Return result
            # -------------------------------------------------

            return {
                "status": "completed",
                "document_id": document_id,
                "filename": file_path.name,
                "documents": len(documents),
                "chunks": len(chunks),
                "stored": stored
            }

        except Exception as error:

            # -------------------------------------------------
            # 11. Mark failed processing
            # -------------------------------------------------

            self.database.update_processing_status(
                document_id=document_id,
                processing_status="failed"
            )

            print(
                f"Processing failed: "
                f"{error}"
            )

            raise