from PIL import Image


class ImagePreprocessor:

    def __init__(
        self,
        max_size=1600
    ):
        self.max_size = max_size

    def convert_rgb(
        self,
        image
    ):

        if image.mode != "RGB":

            return image.convert(
                "RGB"
            )

        return image

    def resize(
        self,
        image
    ):

        image = image.copy()

        width, height = image.size

        if (
            width <= self.max_size
            and height <= self.max_size
        ):
            return image

        scale = min(
            self.max_size / width,
            self.max_size / height
        )

        new_size = (
            int(width * scale),
            int(height * scale)
        )

        return image.resize(
            new_size,
            Image.LANCZOS
        )

    def process(
        self,
        image
    ):

        image = self.convert_rgb(
            image
        )

        image = self.resize(
            image
        )

        return image