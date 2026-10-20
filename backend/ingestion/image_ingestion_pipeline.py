from backend.ingestion.image_foundation_pipeline import ImageFoundationPipeline
from backend.processing.image_content_processor import ImageContentProcessor
from backend.processing.image_chunker import ImageChunker


class ImageIngestionPipeline:
    """
    Complete image ingestion pipeline:

    Image
      -> Foundation
      -> Content processing
      -> Chunking
    """

    def __init__(self):
        self.foundation = ImageFoundationPipeline()
        self.content_processor = ImageContentProcessor()
        self.chunker = ImageChunker()

    def process(self, image_path):
        foundation_result = self.foundation.process(image_path)

        content_result = self.content_processor.process(foundation_result)

        chunks = self.chunker.chunk(content_result)

        return {
            "image_path": str(image_path),
            "foundation": foundation_result,
            "content": content_result,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }

