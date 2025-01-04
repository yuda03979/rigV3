from typing import Literal
from models_dir.model import Model
from globals_dir.utils import handle_errors


class ModelsManager:

    def __init__(self):
        self.models = dict()

    def infer(self, model_nickname, query):
        if not model_nickname in self.models.keys():
            handle_errors(e=f"model_nickname: {model_nickname} do not exist. \nexisting nicknames: {list(self.models.keys())}")
        return self.models[model_nickname].infer(query)

    def load(
            self,
            model_nickname: str,
            *,
            engine: Literal["ollama", "openai"],
            model_name: str,
            task: Literal["generate", "embed"]
    ):
        if model_nickname in self.models.keys():
            # here need to unload the previous model from the RAM
            print(f"model_nickname {model_nickname} already exist. overwriting...")
        self.models[model_nickname] = Model(engine=engine, model_name=model_name, task=task)

    def unload(self, model_nickname):
        if model_nickname in self.models.keys():
            print(f"model_nickname {model_nickname} already exist. no action execute.")
            return
        self.models[model_nickname].unload()

    def get_num_models(self) -> int:
        """
        for managing dynamically the nicknames of models
        :return:
        """
        return len(self.models)

    def __setitem__(self, nickname, value: dict | list | slice):
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
        if not model_nickname in self.models.keys():
            handle_errors(e=f"model_nickname: {model_nickname} do not exist. \nexisting nicknames: {list(self.models.keys())}")
        return self.models.get(model_nickname)
