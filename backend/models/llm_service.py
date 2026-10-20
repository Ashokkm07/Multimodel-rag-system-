from backend.models.model_config import ModelConfig
from backend.models.model_factory import ModelFactory

class LLMService:
    """
    Central LLM service used by the RAG application.
    """

   
    def __init__(
        self,
        provider=None,
        model_name=None,
    ):
        provider = (
            provider
            or ModelConfig.get_provider()
        )

        model_name = (
            model_name
            or ModelConfig.get_model_name()
        )

        self.provider_name = provider
        self.model_name = model_name

        self.provider = ModelFactory.create(
            provider=provider,
            model_name=model_name,
        )

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        return self.provider.generate(
            prompt
        )
  
