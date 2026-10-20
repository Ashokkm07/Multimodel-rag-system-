from PIL import Image
import pytesseract


class ImageOCR:

    def __init__(self):
        self.engine = pytesseract

    def is_available(self):
        return self.engine is not None

    def extract_text(self, image):

        try:
            text = self.engine.image_to_string(
                image
            )

            return {
                "text": text.strip(),
                "status": "success",
                "error": None
            }

        except Exception as error:

            return {
                "text": "",
                "status": "failed",
                "error": str(error)
            }

    def extract_from_file(self, image_path):

        image = Image.open(
            image_path
        )

        image.load()

        return self.extract_text(
            image
        )