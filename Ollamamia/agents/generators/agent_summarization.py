from Ollamamia.globals_dir.models_manager import MODELS_MANAGER
from Ollamamia.globals_dir.utils import get_dict
from Ollamamia.globals_dir.utils import AgentMessage
from src.globals import GLOBALS
import time


class AgentSummarize:
    description = """simple agent that summarize query and removing redundant words. also translate into english"""

    model_nickname = f"AgentSummarize_{GLOBALS.generation_model_name}"
    engine = "ollama"
    model_name = GLOBALS.generation_model_name  # "gemma-2-2b-it-Q8_0:rig"
    model_type = "gemma2"
    task = "generate"

    num_ctx = 2048
    temperature = 0.0
    top_p = 1.0
    stop = ["}"]


    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_{self.model_nickname}"
        self.prompt = ("“Please take the following text and remove redundant words like ua, kind of, i think, "
                       "etc. the message should be clear and strait forward. "
                       "if the text is not in English, "
                       "translate it into English. the text:")

        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.options.num_ctx = self.num_ctx
        MODELS_MANAGER[self.model_nickname].config.options.sop = self.stop
        MODELS_MANAGER[self.model_nickname].config.options.temperature = self.temperature
        MODELS_MANAGER[self.model_nickname].config.options.top_p = self.top_p

    def predict(
            self,
            query
    ):
        query: str = query

        start = time.time()

        prompt = self.prompt + query

        response = MODELS_MANAGER[self.model_nickname].infer(prompt)

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=True,
            agent_message=response,
            message_model=response,
            infer_time=time.time() - start
        )
        return agent_message
