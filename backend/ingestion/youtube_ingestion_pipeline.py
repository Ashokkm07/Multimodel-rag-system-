
from backend.ingestion.youtube_loader import YouTubeLoader
from backend.processing.youtube_processor import process_youtube_transcript
from backend.processing.youtube_chunker import chunk_youtube_transcript
from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore
from backend.utils.database import Database
import hashlib


class YouTubeIngestionPipeline:

    def __init__(self):
        """
        Initialize the YouTube RAG ingestion pipeline.
        """

        self.loader = YouTubeLoader()

        self.embedder = Embedder()

        self.store = ChromaStore()

        self.database = Database()

    def calculate_url_hash(self, youtube_url):
        """
        Generate a SHA-256 hash from the YouTube URL.
        """

        return hashlib.sha256(
            youtube_url.strip().encode("utf-8")
        ).hexdigest()

    def ingest(self, youtube_url):
        """
        Complete YouTube RAG ingestion pipeline.

        YouTube URL
             ↓
        Transcript
             ↓
        Cleaning
             ↓
        Chunking
             ↓
        Embeddings
             ↓
        ChromaDB
             ↓
        SQLite
        """

        youtube_url = youtube_url.strip()

        if not youtube_url:
            raise ValueError(
                "YouTube URL cannot be empty."
            )

        print(
            f"\nLoading YouTube video: {youtube_url}"
        )

        # -------------------------------------------------
        # 1. Extract video ID
        # -------------------------------------------------

        video_id = self.loader.extract_video_id(
            youtube_url
        )

        print(
            f"Video ID: {video_id}"
        )

        # -------------------------------------------------
        # 2. Calculate URL hash
        # -------------------------------------------------

        file_hash = self.calculate_url_hash(
            youtube_url
        )

        print(
            f"URL hash: {file_hash}"
        )

        # -------------------------------------------------
        # 3. Check duplicate
        # -------------------------------------------------

        duplicate = self.database.is_duplicate(
            file_hash
        )

        if duplicate["exists"]:

            existing_status = duplicate[
                "processing_status"
            ]

            print(
                f"\nExisting YouTube source found: "
                f"{duplicate['filename']}"
            )

            print(
                f"Document ID: {duplicate['id']}"
            )

            print(
                f"Existing status: {existing_status}"
            )

            if existing_status == "completed":

                print(
                    "\nDuplicate YouTube video detected."
                )

                return {
                    "status": "duplicate",
                    "document_id": duplicate["id"],
                    "filename": duplicate["filename"],
                    "video_id": video_id,
                    "documents": 0,
                    "chunks": 0,
                    "stored": 0
                }

            elif existing_status == "failed":

                print(
                    "\nPrevious YouTube processing failed."
                )

                print(
                    "Retrying YouTube processing..."
                )

                document_id = duplicate["id"]

                self.database.update_processing_status(
                    document_id=document_id,
                    processing_status="processing",
                    chunk_count=0
                )

            elif existing_status == "processing":

                print(
                    "\nYouTube video is currently "
                    "being processed."
                )

                return {
                    "status": "processing",
                    "document_id": duplicate["id"],
                    "filename": duplicate["filename"],
                    "video_id": video_id,
                    "documents": 0,
                    "chunks": 0,
                    "stored": 0
                }

            else:

                print(
                    "\nUnknown processing status."
                )

                return {
                    "status": "unknown",
                    "document_id": duplicate["id"],
                    "filename": duplicate["filename"],
                    "video_id": video_id,
                    "documents": 0,
                    "chunks": 0,
                    "stored": 0
                }

        else:

            print(
                "\nNew YouTube video detected."
            )

            # -------------------------------------------------
            # 4. Register YouTube source
            # -------------------------------------------------

            document_id = self.database.add_document(
                filename=f"YouTube_{video_id}",
                file_path=youtube_url,
                file_hash=file_hash,
                file_type="youtube",
                processing_status="processing",
                chunk_count=0
            )

            print(
                f"YouTube source registered. "
                f"ID: {document_id}"
            )

        try:

            # -------------------------------------------------
            # 5. Extract transcript
            # -------------------------------------------------

            documents = self.loader.load(
                youtube_url
            )

            print(
                f"Transcript segments extracted: "
                f"{len(documents)}"
            )

            if not documents:

                self.database.update_processing_status(
                    document_id=document_id,
                    processing_status="completed",
                    chunk_count=0
                )

                return {
                    "status": "empty",
                    "document_id": document_id,
                    "filename": f"YouTube_{video_id}",
                    "video_id": video_id,
                    "documents": 0,
                    "chunks": 0,
                    "stored": 0
                }

            # -------------------------------------------------
            # 6. Clean transcript
            # -------------------------------------------------

            processed_documents = (
                process_youtube_transcript(
                    documents
                )
            )

            print(
                f"Transcript segments after cleaning: "
                f"{len(processed_documents)}"
            )

            # -------------------------------------------------
            # 7. Chunk transcript
            # -------------------------------------------------

            chunks = chunk_youtube_transcript(
                processed_documents
            )

            print(
                f"YouTube chunks created: "
                f"{len(chunks)}"
            )

            if not chunks:

                self.database.update_processing_status(
                    document_id=document_id,
                    processing_status="completed",
                    chunk_count=0
                )

                return {
                    "status": "empty",
                    "document_id": document_id,
                    "filename": f"YouTube_{video_id}",
                    "video_id": video_id,
                    "documents": len(documents),
                    "chunks": 0,
                    "stored": 0
                }

            # -------------------------------------------------
            # 8. Add document ID to chunk metadata
            # -------------------------------------------------

            for chunk in chunks:

                chunk["metadata"][
                    "document_id"
                ] = str(document_id)

                chunk["metadata"][
                    "source_type"
                ] = "youtube"

                chunk["metadata"][
                    "video_id"
                ] = video_id

                chunk["metadata"][
                    "source"
                ] = youtube_url

            # -------------------------------------------------
            # 9. Generate embeddings
            # -------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            print(
                "Generating YouTube embeddings..."
            )

            embeddings = self.embedder.embed_documents(
                texts
            )

            print(
                f"Embeddings generated: "
                f"{len(embeddings)}"
            )

            # -------------------------------------------------
            # 10. Store in ChromaDB
            # -------------------------------------------------

            print(
                "Storing YouTube chunks in ChromaDB..."
            )

            stored = self.store.add_chunks(
                chunks,
                embeddings
            )

            print(
                f"YouTube chunks stored: "
                f"{stored}"
            )

            # -------------------------------------------------
            # 11. Update SQLite
            # -------------------------------------------------

            self.database.update_processing_status(
                document_id=document_id,
                processing_status="completed",
                chunk_count=stored
            )

            print(
                "Database status updated: completed"
            )

            return {
                "status": "completed",
                "document_id": document_id,
                "filename": f"YouTube_{video_id}",
                "video_id": video_id,
                "documents": len(documents),
                "chunks": len(chunks),
                "stored": stored
            }

        except Exception as error:

            self.database.update_processing_status(
                document_id=document_id,
                processing_status="failed"
            )

            print(
                f"YouTube processing failed: {error}"
            )

            raise

