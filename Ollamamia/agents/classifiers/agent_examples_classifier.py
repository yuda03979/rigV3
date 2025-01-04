from globals_dir.models_manager import MODELS_MANAGER
from globals_dir.utils import AgentMessage
from agents.logic.basic_rag import BasicRag
import time


class AgentExamplesClassifier:
    description = """rag implemented for elta, suitable for small - medium size db. """

    model_nickname = str(MODELS_MANAGER.get_num_models())
    engine = "ollama"
    model_name = "snowflake-arctic-embed:137m"
    task = "embed"

    similarity_threshold_adding_example: float = 0.5

    max_examples: int = 100_000
    prefix: str = "classification: \n"
    num_examples: int = 2
    softmax: bool = True
    softmax_temperature: float = 0

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_{self.model_nickname}"
        self.basic_rag = BasicRag(model_nickname=self.model_nickname, max_rules=self.num_examples)
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    def predict(self, query: str):
        start = time.time()

        # agent logic
        ####################

        query = self.prefix + query
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query)[0]
        rules_list = self.basic_rag.get_close_types_names(
            query_embedding=query_embeddings,
            softmax=self.softmax,
            temperature=self.softmax_temperature
        )

        example1 = rules_list[0][0]
        example2 = rules_list[1][0]

        ####################

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=True,
            agent_message=[example1, example2],
            message_model=rules_list,
            infer_time=time.time() - start
        )
        return agent_message

    def add_example(self, example: str) -> tuple[str | None, list[float] | None]:
        example = self.prefix + example
        example_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(example)[0]
        other_examples = self.basic_rag.get_close_types_names(
                query_embedding=example_embeddings,
                softmax=self.softmax,
                temperature=self.softmax_temperature
        )
        if other_examples[0][1] > self.similarity_threshold_adding_example:
            print("there's similar examples already. no action perform.")
            return None, None
        else:
            self.basic_rag.add_sample(sample_id=example, sample_embeddings=example_embeddings)
            return example, example_embeddings


    def add_exampleS(self, examples: list[str]) -> tuple[list[str], list[list[float]]]:
        """
        ERASING EXISTING SAMPLES!!
        :param examples_names:
        :param queries_embeddings:
        :return:
        """
        examples_embeddings: list[list[float]] = MODELS_MANAGER[self.model_nickname].infer(examples)
        self.basic_rag.add_samples(samples_ids=examples, sample_embeddings=examples_embeddings)
        return examples, examples_embeddings
