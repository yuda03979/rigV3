from typing import Union, Sequence, Literal
import ollama
from models_dir.ollama_model_config import OllamaModelConfig
from globals_dir.utils import handle_errors


class OllamaBaseModel:

    def __init__(
            self,
            model_name: str,
            task: Literal["embed", "generate"],
    ):
        if model_name not in [info["model"] for info in ollama.list().model_dump()["models"]]:
            handle_errors(e="error occurred - model don't exist")
        self.config = OllamaModelConfig(name=model_name, task=task)

    def _embed(self, query: Union[str, Sequence[str]]) -> list[list]:
        # add prefix to embedding
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

    def infer(self, query):
        if self.config.task == "embed":
            return self._embed(query=query)
        elif self.config.task == "generate":
            return self._generate(query=query)
