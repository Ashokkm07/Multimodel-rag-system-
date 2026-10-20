import hashlib


def calculate_file_hash(file_path, chunk_size=1024 * 1024):
    """
    Calculate the SHA-256 hash of a file.

    The file is read in small chunks so that large files
    do not need to be loaded completely into memory.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:
            data = file.read(chunk_size)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()