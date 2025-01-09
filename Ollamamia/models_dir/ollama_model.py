from typing import Union, Sequence, Literal
import ollama
from ollama import AsyncClient
from Ollamamia.models_dir.async_base import AsyncMixin
from Ollamamia.models_dir.ollama_model_config import OllamaModelConfig
from Ollamamia.globals_dir.utils import handle_errors


class OllamaBaseModel:
    """
    Base class for interacting with Ollama models, providing synchronous embedding and text generation capabilities.

    This class serves as a foundation for working with Ollama models, handling both text embedding
    and text generation tasks. It validates model availability and manages model configurations.

    Args:
        model_name (str): Name of the Ollama model to use
        task (Literal["embed", "generate"]): The task type - either "embed" for embeddings or "generate" for text generation

    Raises:
        Exception: If the specified model doesn't exist in the available Ollama models

    Examples:
        >>> model = OllamaBaseModel(model_name="llama2", task="generate")
        >>> response = model.infer("Write a story about a cat")

        >>> embedder = OllamaBaseModel(model_name="llama2", task="embed")
        >>> embeddings = embedder.infer(["text1", "text2"])
    """

    def __init__(
            self,
            model_name: str,
            task: Literal["embed", "generate"],
    ):
        if model_name not in [info["model"] for info in ollama.list().model_dump()["models"]]:
            handle_errors(e="error occurred - model don't exist")
        self.config = OllamaModelConfig(name=model_name, task=task)

    def _embed(self, query: Union[str, Sequence[str]]) -> list[list]:
        """
        Internal method to generate embeddings for the input text(s).

        Args:
            query (Union[str, Sequence[str]]): Input text or sequence of texts to embed

        Returns:
            list[list]: List of embedding vectors. For single input, returns a single embedding.
                       For sequence input, returns a list of embeddings.

        Note:
            Automatically adds the configured prefix to each input text before embedding.
        """
        if isinstance(query, str):
            query += self.config.prefix
        elif isinstance(query, Sequence):
            query = [q + self.config.prefix for q in query]

        response = ollama.embed(
            model=self.config.name,
            input=query,
            truncate=self.config.truncate,
            options=self.config.options.model_dump(mode="python"),
            keep_alive=self.config.keep_alive,
        )
        return response['embeddings']

    def _generate(self, query: str) -> str:
        """
        Internal method to generate text based on the input prompt.

        Args:
            query (str): Input prompt for text generation

        Returns:
            str: Generated text response
        """
        response = ollama.generate(
            model=self.config.name,
            prompt=query,
            suffix=self.config.suffix,
            system=self.config.system,
            template=self.config.template,
            context=self.config.context,
            raw=self.config.raw,
            format=self.config.format,
            keep_alive=self.config.keep_alive,
            options=self.config.options.model_dump(mode="python")
        )
        return response['response']

    def infer(self, query: Union[str, Sequence[str]]) -> Union[str, list[list]]:
        """
        Main inference method that handles both embedding and generation tasks.

        Args:
            query: For embeddings: str or sequence of str to embed
                  For generation: str prompt for text generation

        Returns:
            For embeddings: list[list] of embedding vectors
            For generation: str containing the generated text

        Raises:
            ValueError: If task type doesn't match the query format
        """
        if self.config.task == "embed":
            return self._embed(query=query)
        elif self.config.task == "generate":
            return self._generate(query=query)


class AsyncOllamaBaseModel(OllamaBaseModel, AsyncMixin):
    """
    Asynchronous version of OllamaBaseModel, providing async embedding and text generation capabilities.

    This class extends OllamaBaseModel to provide asynchronous operations while maintaining
    backward compatibility with synchronous methods. It uses Ollama's AsyncClient for
    asynchronous operations.

    Args:
        model_name (str): Name of the Ollama model to use
        task (Literal["embed", "generate"]): The task type - either "embed" for embeddings or "generate" for text generation

    Examples:
        model = AsyncOllamaBaseModel(model_name="llama2", task="generate")
        response = await model.infer_async("Write a story about a cat")

        embedder = AsyncOllamaBaseModel(model_name="llama2", task="embed")
        embeddings = await embedder.infer_async(["text1", "text2"])
    """

    def __init__(self, model_name: str, task: Literal["embed", "generate"]):
        super().__init__(model_name, task)
        self.async_client = AsyncClient()

    async def _async_embed(self, query: Union[str, Sequence[str]]) -> list[list]:
        """
        Internal async method to generate embeddings for the input text(s).

        Args:
            query (Union[str, Sequence[str]]): Input text or sequence of texts to embed

        Returns:
            list[list]: List of embedding vectors. For single input, returns a single embedding.
                       For sequence input, returns a list of embeddings.
        """
        if isinstance(query, str):
            query += self.config.prefix
        elif isinstance(query, Sequence):
            query = [q + self.config.prefix for q in query]

        response = await self.async_client.embeddings(
            model=self.config.name,
            prompt=query,
            options={
                "truncate": self.config.truncate,
                **self.config.options.model_dump(mode="python")
            }
        )
        return response['embeddings']

    async def _async_generate(self, query: str) -> str:
        """
        Internal async method to generate text based on the input prompt.

        Args:
            query (str): Input prompt for text generation

        Returns:
            str: Generated text response
        """
        response = await self.async_client.generate(
            model=self.config.name,
            prompt=query,
            system=self.config.system,
            template=self.config.template,
            context=self.config.context,
            raw=self.config.raw,
            format=self.config.format,
            options=self.config.options.model_dump(mode="python")
        )
        return response['response']

    async def infer_async(self, query: Union[str, Sequence[str]]) -> Union[str, list[list]]:
        """
        Main asynchronous inference method that handles both embedding and generation tasks.

        Args:
            query: For embeddings: str or sequence of str to embed
                  For generation: str prompt for text generation

        Returns:
            For embeddings: list[list] of embedding vectors
            For generation: str containing the generated text

        Raises:
            ValueError: If task type doesn't match the query format
        """
        if self.config.task == "embed":
            return await self._async_embed(query)
        return await self._async_generate(query)

    def infer(self, query):
        """
        Synchronous inference method maintained for backward compatibility.

        This method calls the parent class's synchronous implementation.
        For asynchronous operations, use infer_async() instead.
        """
        return super().infer(query)
