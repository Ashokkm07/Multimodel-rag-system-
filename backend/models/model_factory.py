from backend.models.gemini_provider import GeminiProvider
from backend.models.gpt_provider import GPTProvider
from backend.models.ollama_provider import OllamaProvider
from backend.models.openrouter_provider import OpenRouterProvider


class ModelFactory:
    """
    Creates the configured LLM provider.
    """

    SUPPORTED_PROVIDERS = {
        "ollama",
        "openrouter",
        "gemini",
        "gpt",
    }

    @classmethod
    def create(cls, provider="ollama", model_name=None):
        if not provider:
            provider = "ollama"

        provider = provider.lower().strip()

        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if provider == "ollama":
            return OllamaProvider(model_name=model_name)

        if provider == "openrouter":
            return OpenRouterProvider(model_name=model_name)

        if provider == "gemini":
            return GeminiProvider(model_name=model_name)

        if provider == "gpt":
            return GPTProvider(model_name=model_name)

        raise RuntimeError("Provider factory reached an invalid state.")