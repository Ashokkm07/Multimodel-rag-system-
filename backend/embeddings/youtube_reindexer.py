from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


class YouTubeReindexer:

    def __init__(self):

        self.embedder = Embedder()

        self.store = ChromaStore()

    def reindex_video(
        self,
        video_id,
        chunks
    ):
        """
        Reindex one YouTube video.

        Existing chunks belonging only to this video
        are deleted first.

        The supplied chunks should already be processed
        timestamp-aware YouTube chunks.
        """

        if not video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        print(
            f"\nReindexing YouTube video: "
            f"{video_id}"
        )

        # -------------------------------------------------
        # 1. Delete old chunks
        # -------------------------------------------------

        print(
            "\nDeleting old chunks for this video..."
        )

        self.store.delete_youtube_video(
            video_id
        )

        print(
            "Old video chunks deleted."
        )

        # -------------------------------------------------
        # 2. Validate chunks
        # -------------------------------------------------

        if not chunks:

            print(
                "No chunks available for reindexing."
            )

            return {
                "status": "empty",
                "video_id": video_id,
                "chunks": 0,
                "stored": 0
            }

        # -------------------------------------------------
        # 3. Ensure metadata is correct
        # -------------------------------------------------

        valid_chunks = []

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {}
            )

            chunk_video_id = str(
                metadata.get(
                    "video_id",
                    ""
                )
            )

            if chunk_video_id != str(
                video_id
            ):

                continue

            metadata["source_type"] = (
                "youtube"
            )

            metadata["video_id"] = (
                str(video_id)
            )

            chunk["metadata"] = metadata

            valid_chunks.append(
                chunk
            )

        if not valid_chunks:

            return {
                "status": "invalid",
                "video_id": video_id,
                "chunks": 0,
                "stored": 0
            }

        print(
            f"Valid chunks: "
            f"{len(valid_chunks)}"
        )

        # -------------------------------------------------
        # 4. Generate embeddings
        # -------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in valid_chunks
        ]

        print(
            "\nGenerating embeddings..."
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

        # -------------------------------------------------
        # 5. Store chunks
        # -------------------------------------------------

        print(
            "\nStoring new YouTube chunks..."
        )

        stored = self.store.add_chunks(
            valid_chunks,
            embeddings
        )

        print(
            f"New chunks stored: "
            f"{stored}"
        )

        return {
            "status": "completed",
            "video_id": video_id,
            "chunks": len(valid_chunks),
            "stored": stored
        }

    def get_total_count(self):

        return self.store.count()