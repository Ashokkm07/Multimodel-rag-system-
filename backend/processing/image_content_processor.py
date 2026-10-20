class ImageContentProcessor:
    """
    Combines OCR text and vision-model output into a normalized
    representation suitable for later chunking and embedding.
    """

    def process(self, foundation_result):
        if not foundation_result:
            raise ValueError("Image foundation result cannot be empty.")

        metadata = foundation_result.get("metadata", {})

        ocr_text = foundation_result.get("ocr_text", "").strip()

        vision_description = foundation_result.get(
            "vision_description", ""
        ).strip()

        content_parts = []

        if ocr_text:
            content_parts.append(
                "OCR TEXT:\n" + ocr_text
            )

        if vision_description:
            content_parts.append(
                "VISION DESCRIPTION:\n" + vision_description
            )

        if content_parts:
            combined_text = "\n\n".join(content_parts)
        else:
            combined_text = (
                "No readable OCR text or vision description "
                "was extracted from the image."
            )

        processed_metadata = {
            **metadata,
            "ocr_available": bool(ocr_text),
            "vision_available": bool(vision_description),
            "ocr_status": foundation_result.get(
                "ocr_status",
                "unknown"
            ),
            "vision_status": foundation_result.get(
                "vision_status",
                "unknown"
            ),
        }

        return {
            "text": combined_text,
            "ocr_text": ocr_text,
            "vision_description": vision_description,
            "metadata": processed_metadata,
        }
