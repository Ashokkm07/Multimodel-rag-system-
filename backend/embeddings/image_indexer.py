import hashlib
from pathlib import Path

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

class ImageIndexer:
    """
    Embeds image knowledge and stores it in the multimodal Chroma collection.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.store = ChromaStore()

    def _image_id(self, image_path):
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        sha256 = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                sha256.update(block)

        return sha256.hexdigest()

    def reindex_image(self, image_path, chunks):
        image_path = str(image_path)
        image_id = self._image_id(image_path)

        valid_chunks = []

        for chunk in chunks or []:
            text = chunk.get("text", "").strip()

            if not text:
                continue

            metadata = dict(chunk.get("metadata", {}))

            metadata["source_type"] = "image"
            metadata["source"] = image_path
            metadata["image_id"] = image_id

            valid_chunks.append(
                {
                    "text": text,
                    "metadata": metadata,
                }
            )

        # Remove previous chunks belonging to this exact image.
        self.store.collection.delete(
            where={
                "$and": [
                    {"source_type": "image"},
                    {"image_id": image_id},
                ]
            }
        )

        if not valid_chunks:
            return {
                "status": "completed",
                "image_id": image_id,
                "chunks": 0,
                "stored": 0,
            }

        texts = [
            chunk["text"]
            for chunk in valid_chunks
        ]

        embeddings = self.embedder.embed_documents(texts)

        stored = self.store.add_chunks(
            valid_chunks,
            embeddings,
        )

        return {
            "status": "completed",
            "image_id": image_id,
            "chunks": len(valid_chunks),
            "stored": stored,
        }

