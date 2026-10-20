from backend.models.base_model import BaseLLM
from backend.models.ollama_model import OllamaModel

class OllamaProvider(BaseLLM):
    """
    Provider adapter for the existing OllamaModel.
    """


    def __init__(self, model_name=None):
        if model_name:
            self.model = OllamaModel(
                model_name=model_name
            )
        else:
            self.model = OllamaModel()

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        return self.model.generate(
            prompt
        )
  
