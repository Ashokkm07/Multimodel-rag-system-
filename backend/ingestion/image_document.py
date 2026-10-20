from pathlib import Path

from backend.ingestion.image_loader import load_image
from backend.processing.image_content_processor import ImageContentProcessor


class ImageDocumentLoader:

    def __init__(self, vision_model_name="llava:7b"):
        """
        Initialize the image content processor.
        """

        self.processor = ImageContentProcessor(
            vision_model_name=vision_model_name
        )

    def load(self, image_path, document_id=None):
        """
        Convert an image into the standard RAG document format.
        """

        image_path = Path(image_path)

        image_info = load_image(image_path)

        content = self.processor.process(
            str(image_path)
        )

        metadata = {
            **image_info["metadata"],
            "source_type": "image"
        }

        if document_id is not None:
            metadata["document_id"] = str(document_id)

        return [{
            "text": content["text"],
            "metadata": metadata
        }]