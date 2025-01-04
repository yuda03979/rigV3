from typing import Literal
from .ollama_model import OllamaBaseModel


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
