from Ollamamia.globals_dir.models_manager import MODELS_MANAGER
from Ollamamia.globals_dir.utils import AgentMessage
from Ollamamia.agents.logic.basic_rag import BasicRag
import time
from src.globals import GLOBALS


class AgentExamplesClassifier:
    """
    A classifier agent that manages and retrieves similar examples using RAG (Retrieval-Augmented Generation).

    This implementation is suitable for small to medium-sized databases and uses embeddings to find
    similar examples. It prevents duplicate examples by checking similarity before addition and
    returns the two most similar examples for each query.

    Attributes:
        description (str): Description of the RAG implementation
        model_nickname (str): Unique identifier for the model instance
        engine (str): The engine used for embeddings ("ollama")
        model_name (str): Name of the model from GLOBALS
        task (str): Type of task the model performs ("embed")
        max_examples (int): Maximum number of examples that can be stored (100,000)
        prefix (str): Prefix added to improve RAG accuracy
        softmax (bool): Whether to apply softmax to similarity scores (False)
        softmax_temperature (float): Temperature parameter for softmax (0)
    """

    description = """rag implemented for elta, suitable for small - medium size db"""

    model_nickname = f"AgentExamplesClassifier_{GLOBALS.rag_model_name}"
    engine = "ollama"
    model_name = GLOBALS.rag_model_name  # "snowflake-arctic-embed:137m"
    task = "embed"

    max_examples: int = 100_000
    prefix: str = "classification: \n"
    softmax: bool = False
    softmax_temperature: float = 0

    def __init__(self, agent_name: str):
        """
        Initialize the AgentExamplesClassifier.

        Args:
            agent_name (str): Name identifier for the agent instance
        """
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_{self.model_nickname}"
        self.basic_rag = BasicRag(max_samples=self.max_examples)
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    @property
    def similarity_threshold_adding_example(self):
        """
        Get the similarity threshold for adding new examples from globals.

        Returns:
            float: Minimum similarity threshold that determines if an example is too similar
                  to existing ones to be added
        """
        return GLOBALS.examples_rag_threshold  # without softmax

    def predict(self, query: str) -> AgentMessage:
        """
        Find the two most similar examples to the given query.

        Args:
            query (str): The text to find similar examples for

        Returns:
            AgentMessage: A message object containing:
                - agent_name: Name of the classifier agent
                - agent_description: Description of the agent
                - agent_input: Original query
                - succeed: Always True
                - agent_message: List of two most similar examples [example1, example2]
                - message_model: List of two (example, similarity_score) tuples
                - infer_time: Time taken for prediction
        """
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
            examples_list = []
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
        """
        Add a single example if it's not too similar to existing ones.

        Args:
            example (str): The example text to potentially add

        Returns:
            tuple containing:
                - success (bool): Whether the example was added successfully
                - index (int | None): Index of the example if it already existed
                - example (str | None): The example text
                - example_embeddings (list[float] | None): The embedded representation

        Note:
            The example will not be added if its similarity to any existing example
            exceeds similarity_threshold_adding_example
        """
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
        Add multiple examples at once.

        WARNING: This method erases all existing examples before adding new ones!

        Args:
            examples (list[str]): List of example texts to add

        Returns:
            tuple containing:
                - examples (list[str]): List of added example texts
                - examples_embeddings (list[list[float]]): List of embedded representations
        """
        examples_embeddings: list[list[float]] = MODELS_MANAGER[self.model_nickname].infer(examples)
        self.basic_rag.add_samples(samples_ids=examples, sample_embeddings=examples_embeddings)
        return examples, examples_embeddings

    def add_embedded_examples(self, examples: list[str], embedded_examples: list[list[float]]) -> None:
        """
        Add pre-embedded examples to the classifier.

        Args:
            examples (list[str]): List of example texts
            embedded_examples (list[list[float]]): List of pre-computed embeddings

        Note:
            This method assumes the embeddings were generated using the same model
            and configuration as the current classifier.
        """
        self.basic_rag.add_samples(samples_ids=examples, sample_embeddings=embedded_examples)
