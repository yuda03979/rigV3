from Ollamamia.globals_dir.models_manager import MODELS_MANAGER
from Ollamamia.globals_dir.utils import AgentMessage
from Ollamamia.agents.logic.basic_rag import BasicRag
import time
from src.globals import GLOBALS

class AgentExamplesClassifier:
    description = """rag implemented for elta, suitable for small - medium size db"""

    model_nickname = str(MODELS_MANAGER.get_num_models())
    engine = "ollama"
    model_name = GLOBALS.rag_model_name  # "snowflake-arctic-embed:137m"
    task = "embed"

    max_examples: int = 100_000
    prefix: str = "classification: \n"
    num_examples: int = 100_000
    softmax: bool = False
    softmax_temperature: float = 0

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_{self.model_nickname}"
        self.basic_rag = BasicRag(max_rules=self.num_examples)
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    @property
    def similarity_threshold_adding_example(self):
        return GLOBALS.examples_rag_threshold  # without softmax

    def predict(self, query: str):
        start = time.time()

        # agent logic
        ####################

        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query)[0]
        examples_list = self.basic_rag.get_close_types_names(
            query_embedding=query_embeddings,
            softmax=self.softmax,
            temperature=self.softmax_temperature
        )

        if not examples_list or len(examples_list) < 2:
            example1 = None
            example2 = None
        else:
            example1 = examples_list[0][0]
            example2 = examples_list[1][0]

        ####################

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=True,
            agent_message=[example1, example2],
            message_model=examples_list[:2],
            infer_time=time.time() - start
        )
        return agent_message

    def add_example(self, example: str) -> tuple[bool, int | None, str | None, list[float] | None]:
        example_embeddings: list[float] | None = MODELS_MANAGER[self.model_nickname].infer(example)[0]
        other_examples = self.basic_rag.get_close_types_names(
            query_embedding=example_embeddings,
            softmax=False,
            temperature=self.softmax_temperature
        )

        if not other_examples or not other_examples[0][1] > self.similarity_threshold_adding_example:
            success, index = self.basic_rag.add_sample(sample_id=example, sample_embeddings=example_embeddings)
            return success, index, example, example_embeddings
        else:
            # print("there's similar examples already. no action perform.")
            return False, None, example, example_embeddings

    def add_exampleS(self, examples: list[str]) -> tuple[list[str], list[list[float]]]:
        """
        ERASING EXISTING SAMPLES!!
        :param examples:
        :return:
        """
        examples_embeddings: list[list[float]] = MODELS_MANAGER[self.model_nickname].infer(examples)
        self.basic_rag.add_samples(samples_ids=examples, sample_embeddings=examples_embeddings)
        return examples, examples_embeddings

    def add_embedded_examples(self, examples: list[str], embedded_examples: list[list[float]]) -> None:
        self.basic_rag.add_samples(samples_ids=examples, sample_embeddings=embedded_examples)