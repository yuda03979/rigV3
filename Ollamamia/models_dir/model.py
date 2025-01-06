from typing import Literal

from Ollamamia.models_dir.async_base import AsyncMixin
from Ollamamia.models_dir.ollama_model import OllamaBaseModel, AsyncOllamaBaseModel


class Model:

    def __init__(self, engine: Literal["ollama", "openai"], model_name: str, task: Literal["generate", "embed"]):
        self.model = None
        self.config = None
        self.engine = engine
        self.model_name = model_name
        self.task = task

        self.len_logs = 1000
        self.logs = []

        self.init_model()

    def init_model(self):
        if self.engine == "ollama":
            self.model = OllamaBaseModel(model_name=self.model_name, task=self.task)
            self.config = self.model.config
        elif self.engine == "openai":
            pass

    def infer(self, query) -> str | list:
        response = self.model.infer(query)
        self._manage_logs(response=response)
        return response

    def _manage_logs(self, response):
        if len(self.logs) >= self.len_logs:
            self.logs.pop(0)
        self.logs.append(response)

    def unload(self):
        pass



class AsyncModel(Model, AsyncMixin):
    def __init__(self, engine: Literal["ollama", "openai"], model_name: str, task: Literal["generate", "embed"]):
        super().__init__(engine, model_name, task)

    def init_model(self):
        if self.engine == "ollama":
            self.model = AsyncOllamaBaseModel(model_name=self.model_name, task=self.task)
            self.config = self.model.config
        elif self.engine == "openai":
            pass

    async def infer_async(self, query):
        if hasattr(self.model, 'infer_async'):
            response = await self.model.infer_async(query)
        else:
            response = await self._run_async(self.model.infer, query)
        self._manage_logs(response)
        return response
