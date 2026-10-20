from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Common interface for every LLM provider.
    """

   
    @abstractmethod
    def generate(self, prompt):
        """
        Generate text from a prompt.
        """
        raise NotImplementedError
   
