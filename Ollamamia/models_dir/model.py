from typing import Literal

from Ollamamia.models_dir.async_base import AsyncMixin
from Ollamamia.models_dir.ollama_model import OllamaBaseModel, AsyncOllamaBaseModel


class Model:
    """
    A unified interface for different AI model engines (currently supports Ollama, with OpenAI planned).

    This class provides a consistent way to interact with different AI models for both text generation
    and embedding tasks. It includes built-in logging of model responses and handles model initialization.

    Args:
        engine (Literal["ollama", "openai"]): The AI engine to use ("openai" support pending)
        model_name (str): Name of the model to load (e.g., "llama2" for Ollama)
        task (Literal["generate", "embed"]): Type of task - text generation or embedding

    Attributes:
        model: The underlying model instance
        config: Configuration for the current model
        logs (list): History of model responses, limited by len_logs
        len_logs (int): Maximum number of responses to keep in history (default: 1000)

    Examples:
        model = Model(engine="ollama", model_name="llama2", task="generate")
        response = model.infer("Write a story about a cat")

        embedder = Model(engine="ollama", model_name="llama2", task="embed")
        embeddings = embedder.infer(["text1", "text2"])
    """

    def __init__(self, engine: Literal["ollama", "openai"], model_name: str, task: Literal["generate", "embed"]):
        self.model = None
        self.config = None
        self.engine = engine
        self.model_name = model_name
        self.task = task

        self.len_logs = 100
        self.logs = []

        self.init_model()

    def init_model(self):
        """
        Initializes the appropriate model based on the selected engine.

        Currently supports Ollama models, with OpenAI support planned for future implementation.
        """
        if self.engine == "ollama":
            self.model = OllamaBaseModel(model_name=self.model_name, task=self.task)
            self.config = self.model.config
        elif self.engine == "openai":
            pass

    def infer(self, query) -> str | list:
        """
        Performs inference using the initialized model.

        Args:
            query: Input for the model:
                  - For generation tasks: str containing the prompt
                  - For embedding tasks: str or sequence of str to embed

        Returns:
            str | list: Model's response:
                       - For generation tasks: str containing generated text
                       - For embedding tasks: list of embedding vectors

        Note:
            Each response is automatically logged in the model's history.
        """
        response = self.model.infer(query)
        self._manage_logs(response=response)
        return response

    def _manage_logs(self, response):
        """
        Manages the response history log with a fixed-size FIFO queue.

        Args:
            response: The model's response to log

        Note:
            Maintains at most len_logs entries, removing oldest entries when full.
        """
        if len(self.logs) >= self.len_logs:
            self.logs.pop(0)
        self.logs.append(response)

    def unload(self):
        """
        Cleanup method for model resources.

        Currently a placeholder for future implementation of cleanup operations.
        """
        pass


class AsyncModel(Model, AsyncMixin):
    """
    Asynchronous version of the Model class, supporting async operations for supported engines.

    This class extends the base Model class with asynchronous capabilities while maintaining
    all the same functionality. It can handle both async-native models and wrap synchronous
    models in async operations.

    Args:
        engine (Literal["ollama", "openai"]): The AI engine to use ("openai" support pending)
        model_name (str): Name of the model to load (e.g., "llama2" for Ollama)
        task (Literal["generate", "embed"]): Type of task - text generation or embedding

    Examples:
        >>> model = AsyncModel(engine="ollama", model_name="llama2", task="generate")
        >>> response = await model.infer_async("Write a story about a cat")

        >>> embedder = AsyncModel(engine="ollama", model_name="llama2", task="embed")
        >>> embeddings = await embedder.infer_async(["text1", "text2"])
    """

    def __init__(self, engine: Literal["ollama", "openai"], model_name: str, task: Literal["generate", "embed"]):
        super().__init__(engine, model_name, task)

    def init_model(self):
        """
        Initializes the appropriate async model based on the selected engine.

        Currently supports async Ollama models, with OpenAI support planned for future implementation.
        """
        if self.engine == "ollama":
            self.model = AsyncOllamaBaseModel(model_name=self.model_name, task=self.task)
            self.config = self.model.config
        elif self.engine == "openai":
            pass

    async def infer_async(self, query):
        """
        Performs asynchronous inference using the initialized model.

        If the underlying model supports async operations directly (has infer_async method),
        uses that. Otherwise, wraps the synchronous inference in an async operation.

        Args:
            query: Input for the model:
                  - For generation tasks: str containing the prompt
                  - For embedding tasks: str or sequence of str to embed

        Returns:
            The model's response:
            - For generation tasks: str containing generated text
            - For embedding tasks: list of embedding vectors

        Note:
            Each response is automatically logged in the model's history.
        """
        if hasattr(self.model, 'infer_async'):
            response = await self.model.infer_async(query)
        else:
            response = await self._run_async(self.model.infer, query)
        self._manage_logs(response)
        return response