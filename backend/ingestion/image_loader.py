from pathlib import Path

from PIL import Image


class ImageLoader:

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif"
    }

    def __init__(self):
        pass

    def validate(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Path is not a file: {image_path}"
            )

        extension = (
            image_path.suffix.lower()
        )

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {extension}"
            )

        return True

    def load(self, image_path):

        image_path = Path(image_path)

        self.validate(
            image_path
        )

        try:

            image = Image.open(
                image_path
            )

            image.load()

            return image

        except Exception as error:

            raise RuntimeError(
                f"Could not load image: {error}"
            )

    def get_metadata(
        self,
        image_path,
        image
    ):

        image_path = Path(
            image_path
        )

        return {
            "source": str(
                image_path
            ),
            "source_type": "image",
            "filename": image_path.name,
            "extension": image_path.suffix.lower(),
            "width": image.width,
            "height": image.height,
            "mode": image.mode
        }