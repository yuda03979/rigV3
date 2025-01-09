from typing import Literal

from Ollamamia.models_dir.async_base import AsyncMixin
from Ollamamia.models_dir.model import Model, AsyncModel
from Ollamamia.globals_dir.utils import handle_errors


class ModelsManager:
    """
    A manager class for handling multiple AI models and their lifecycle.

    Provides a centralized way to load, unload, and interact with multiple AI models.
    Supports different model engines (currently Ollama and OpenAI) and different tasks
    (generation and embedding). Models are referenced by nicknames for easy access.

    Examples:
        manager = ModelsManager()
        manager.load("gpt", engine="ollama", model_name="llama2", task="generate")
        response = manager.infer("gpt", "Write a story")

        # Alternative dictionary-style loading:
        manager["gpt"] = {"engine": "ollama", "model_name": "llama2", "task": "generate"}
        model = manager["gpt"]
    """

    def __init__(self):
        """
        Initializes an empty models manager.
        """
        self.models = dict()

    def infer(self, model_nickname, query):
        """
        Performs inference using a specified model.

        Args:
            model_nickname (str): Nickname of the model to use
            query: Input query for the model (format depends on model type and task)

        Returns:
            Model's response (format depends on task type)

        Raises:
            Exception: If the specified model nickname doesn't exist
        """
        if not model_nickname in self.models.keys():
            handle_errors(
                e=f"model_nickname: {model_nickname} do not exist. \nexisting nicknames: {list(self.models.keys())}")
        return self.models[model_nickname].infer(query)

    def load(
            self,
            model_nickname: str,
            *,
            engine: Literal["ollama", "openai"],
            model_name: str,
            task: Literal["generate", "embed"]
    ):
        """
        Loads a new model into the manager.

        Args:
            model_nickname (str): Nickname to reference the model by
            engine (Literal["ollama", "openai"]): AI engine to use
            model_name (str): Name of the model to load
            task (Literal["generate", "embed"]): Task type for the model

        Note:
            If a model with the same nickname exists, it will be overwritten.
            Future implementation may include cleanup of previous model resources.
        """
        if model_nickname in self.models.keys():
            # here need to unload the previous model from the RAM
            print(f"model_nickname {model_nickname} already exist. overwriting...")
        self.models[model_nickname] = Model(engine=engine, model_name=model_name, task=task)

    def unload(self, model_nickname):
        """
        Unloads a model from the manager.

        Args:
            model_nickname (str): Nickname of the model to unload

        Note:
            Current implementation is a placeholder for future resource cleanup.
        """
        if model_nickname in self.models.keys():
            print(f"model_nickname {model_nickname} already exist. no action execute.")
            return
        self.models[model_nickname].unload()

    def get_num_models(self) -> int:
        """
        Returns the number of currently loaded models.

        Returns:
            int: Number of models currently managed
        """
        return len(self.models)

    def __setitem__(self, nickname, value: dict | list | slice):
        """
        Enables dictionary-style model loading with multiple input formats:

        Args:
            nickname (str): Nickname for the model
            value: Model configuration in one of these formats:
                - dict: {"engine": str, "model_name": str, "task": str}
                - list/tuple: [engine, model_name, task]
                - slice: engine:model_name:task

        Examples:
            manager["gpt"] = {"engine": "ollama", "model_name": "llama2", "task": "generate"}
            manager["gpt"] = ["ollama", "llama2", "generate"]
            manager["gpt"] = "ollama":"llama2":"generate"
        """
        if isinstance(value, dict):
            engine = value["engine"]
            model_name = value["model_name"]
            task = value["task"]
        elif isinstance(value, list) or isinstance(value, tuple):
            engine = value[0]
            model_name = value[1]
            task = value[2]
        elif isinstance(value, slice):
            engine = value.start
            model_name = value.stop
            task = value.step
        else:
            raise
        self.load(nickname, engine=engine, model_name=model_name, task=task)

    def __getitem__(self, model_nickname):
        """
        Enables dictionary-style model access.

        Args:
            model_nickname (str): Nickname of the model to retrieve

        Returns:
            Model: The requested model instance

        Raises:
            Exception: If the specified model nickname doesn't exist
        """
        if not model_nickname in self.models.keys():
            handle_errors(
                e=f"model_nickname: {model_nickname} do not exist. \nexisting nicknames: {list(self.models.keys())}")
        return self.models.get(model_nickname)


class AsyncModelsManager(ModelsManager, AsyncMixin):
    """
    Asynchronous version of ModelsManager, supporting async operations.

    Extends ModelsManager to provide asynchronous model operations while maintaining
    all the same functionality. Models loaded through this manager will be async-capable.

    Examples:
        manager = AsyncModelsManager()
        manager.load("gpt", engine="ollama", model_name="llama2", task="generate")
        response = await manager.infer_async("gpt", "Write a story")
    """

    def load(self, model_nickname: str, *, engine: Literal["ollama", "openai"],
             model_name: str, task: Literal["generate", "embed"]):
        """
        Loads a new async-capable model into the manager.

        Args:
            model_nickname (str): Nickname to reference the model by
            engine (Literal["ollama", "openai"]): AI engine to use
            model_name (str): Name of the model to load
            task (Literal["generate", "embed"]): Task type for the model
        """
        if model_nickname in self.models:
            print(f"model_nickname {model_nickname} already exists. overwriting...")
        self.models[model_nickname] = AsyncModel(engine=engine, model_name=model_name, task=task)

    async def infer_async(self, model_nickname, query):
        """
        Performs asynchronous inference using a specified model.

        Args:
            model_nickname (str): Nickname of the model to use
            query: Input query for the model (format depends on model type and task)

        Returns:
            Model's response (format depends on task type)

        Raises:
            Exception: If the specified model nickname doesn't exist
        """
        if model_nickname not in self.models:
            handle_errors(
                f"model_nickname: {model_nickname} doesn't exist. \nexisting nicknames: {list(self.models.keys())}")
        return await self.models[model_nickname].infer_async(query)