
from backend.ingestion.image_loader import (
    ImageLoader
)

from backend.processing.image_preprocessor import (
    ImagePreprocessor
)

from backend.ingestion.image_ocr import (
    ImageOCR
)

from backend.models.vision_model import (
    VisionModel
)


class ImageFoundationPipeline:

    def __init__(self):

        self.loader = ImageLoader()

        self.preprocessor = (
            ImagePreprocessor()
        )

        self.ocr = ImageOCR()

        self.vision = VisionModel()

    def process(
        self,
        image_path
    ):

        # ============================================
        # 1. LOAD IMAGE
        # ============================================

        image = self.loader.load(
            image_path
        )

        metadata = (
            self.loader.get_metadata(
                image_path,
                image
            )
        )

        # ============================================
        # 2. PREPROCESS IMAGE
        # ============================================

        processed_image = (
            self.preprocessor.process(
                image
            )
        )

        # ============================================
        # 3. OCR
        # ============================================

        ocr_result = (
            self.ocr.extract_text(
                processed_image
            )
        )

        # ============================================
        # 4. VISION MODEL
        # ============================================

        vision_prompt = """
Analyze this image carefully.

Describe only what is actually visible.

Include, when applicable:

- objects
- people
- text
- diagrams
- charts
- tables
- visual layout
- important relationships
- visible labels
- important visual details

Do not invent information.

If text is visible but difficult to read,
describe only what can be reliably recognized.
"""

        try:

            vision_description = (
                self.vision.analyze(
                    processed_image,
                    vision_prompt
                )
            )

            vision_status = "success"
            vision_error = None

        except Exception as error:

            vision_description = ""

            vision_status = "failed"

            vision_error = str(
                error
            )

        # ============================================
        # 5. COMBINE OCR + VISION
        # ============================================

        combined_parts = []

        if ocr_result["text"]:

            combined_parts.append(
                "OCR TEXT:\n"
                + ocr_result["text"]
            )

        if vision_description:

            combined_parts.append(
                "VISION DESCRIPTION:\n"
                + vision_description
            )

        if combined_parts:

            combined_text = (
                "\n\n".join(
                    combined_parts
                )
            )

        else:

            combined_text = (
                "No readable OCR text or vision "
                "description was extracted from "
                "the image."
            )

        # ============================================
        # 6. RETURN IMAGE KNOWLEDGE
        # ============================================

        return {

            "text": combined_text,

            "ocr_text": (
                ocr_result["text"]
            ),

            "ocr_status": (
                ocr_result["status"]
            ),

            "ocr_error": (
                ocr_result["error"]
            ),

            "vision_description": (
                vision_description
            ),

            "vision_status": (
                vision_status
            ),

            "vision_error": (
                vision_error
            ),

            "metadata": metadata
        }

