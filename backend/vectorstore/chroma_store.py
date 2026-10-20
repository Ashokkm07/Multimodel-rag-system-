
import hashlib
import chromadb
from pathlib import Path


class ChromaStore:

    def __init__(
        self,
        persist_directory="data/chroma"
    ):
        """
        Initialize the persistent ChromaDB store.
        """

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
                name="multimodal_rag"
            )
        )

    # -------------------------------------------------
    # GENERATE UNIQUE CHUNK ID
    # -------------------------------------------------

    def generate_chunk_id(self, chunk):
        """
        Generate a deterministic unique ID for a chunk.

        The ID uses the source metadata plus the chunk
        text. Different source types can therefore have
        different identifiers even when their text is
        identical.
        """

        metadata = chunk.get(
            "metadata",
            {}
        )

        text = chunk.get(
            "text",
            ""
        )

        source = str(
            metadata.get(
                "source",
                ""
            )
        )

        source_type = str(
            metadata.get(
                "source_type",
                ""
            )
        )

        document_id = str(
            metadata.get(
                "document_id",
                ""
            )
        )

        page = str(
            metadata.get(
                "page",
                ""
            )
        )

        slide = str(
            metadata.get(
                "slide",
                ""
            )
        )

        paragraph = str(
            metadata.get(
                "paragraph",
                ""
            )
        )

        row = str(
            metadata.get(
                "row",
                ""
            )
        )

        # -------------------------------------------------
        # YouTube-specific metadata
        # -------------------------------------------------

        video_id = str(
            metadata.get(
                "video_id",
                ""
            )
        )

        segment = str(
            metadata.get(
                "segment",
                ""
            )
        )

        start_time = str(
            metadata.get(
                "start_time",
                ""
            )
        )

        end_time = str(
            metadata.get(
                "end_time",
                ""
            )
        )

        chunk_number = str(
            metadata.get(
                "chunk",
                ""
            )
        )

        # -------------------------------------------------
        # Build deterministic unique string
        # -------------------------------------------------

        unique_string = "|".join([
            source_type,
            source,
            document_id,
            video_id,
            page,
            slide,
            paragraph,
            row,
            segment,
            start_time,
            end_time,
            chunk_number,
            text
        ])

        hash_value = hashlib.sha256(
            unique_string.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"chunk_{hash_value}"
        )

    # -------------------------------------------------
    # ADD CHUNKS
    # -------------------------------------------------

    def add_chunks(
        self,
        chunks,
        embeddings
    ):
        """
        Add or update chunks in ChromaDB.

        ChromaDB upsert allows existing deterministic
        chunk IDs to be updated safely.
        """

        if len(chunks) != len(embeddings):

            raise ValueError(
                "Number of chunks and embeddings "
                "must be identical."
            )

        ids = []

        documents = []

        metadatas = []

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            chunk_id = (
                self.generate_chunk_id(
                    chunk
                )
            )

            ids.append(
                chunk_id
            )

            documents.append(
                chunk["text"]
            )

            metadata = {}

            for key, value in (
                chunk["metadata"].items()
            ):

                metadata[str(key)] = str(
                    value
                )

            metadatas.append(
                metadata
            )

        # -------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------

        if len(ids) != len(set(ids)):

            raise ValueError(
                "Duplicate chunk IDs were generated "
                "before ChromaDB insertion."
            )

        # -------------------------------------------------
        # UPSERT
        # -------------------------------------------------

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(ids)

    # -------------------------------------------------
    # DELETE YOUTUBE VIDEO CHUNKS
    # -------------------------------------------------

    def delete_youtube_video(
        self,
        video_id
    ):
        """
        Delete all existing YouTube chunks belonging
        to the specified video.

        Only records with:
            source_type = youtube
            video_id = given video_id

        are removed.

        Document, Image, and GitHub chunks are not affected.
        """

        if not video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        self.collection.delete(
            where={
                "$and": [
                    {
                        "source_type": "youtube"
                    },
                    {
                        "video_id": str(
                            video_id
                        )
                    }
                ]
            }
        )

    # -------------------------------------------------
    # COUNT
    # -------------------------------------------------

    def count(self):
        """
        Return the total number of chunks currently
        stored in the ChromaDB collection.
        """

        return self.collection.count()
