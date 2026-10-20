import os

from dotenv import load_dotenv

load_dotenv()


class ModelConfig:
    """
    Reads LLM configuration from the .env file.
    """

    @staticmethod
    def get_provider():
        return os.getenv(
            "LLM_PROVIDER",
            "ollama",
        ).strip().lower()

    @staticmethod
    def get_model_name():
        return os.getenv(
            "LLM_MODEL",
            "llama3.2:3b",
        ).strip()