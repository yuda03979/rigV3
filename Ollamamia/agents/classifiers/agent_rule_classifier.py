import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Ollamamia.globals_dir.models_manager import MODELS_MANAGER
from Ollamamia.globals_dir.utils import AgentMessage
from Ollamamia.agents.logic.basic_rag import BasicRag
from src.globals import GLOBALS
import time


class AgentRuleClassifier:
    """
    A rule-based classifier agent using RAG (Retrieval-Augmented Generation) for text classification.

    This agent is designed for small to medium-sized databases and uses embeddings to classify
    text inputs according to predefined rules. It leverages the Ollama framework for embedding
    generation and supports both single and batch rule additions.

    Attributes:
        description (str): Description of the RAG implementation
        model_nickname (str): Unique identifier for the model instance
        engine (str): The engine used for embeddings ("ollama")
        model_name (str): Name of the model from GLOBALS
        task (str): Type of task the model performs ("embed")
        max_rules (int): Maximum number of rules that can be stored (default: 100,000)
        prefix (str): Prefix added to improve RAG accuracy
        softmax (bool): Whether to apply softmax to similarity scores
    """

    description = """rag implemented for elta, suitable for small - medium size db. """
    engine = "ollama"
    model_name = GLOBALS.rag_model_name  # "snowflake-arctic-embed:137m"

    task = "embed"
    max_rules: int = 100_000
    prefix: str = "classification:\n"
    softmax: bool = True

    def __init__(self, agent_name: str):
        """
        Initialize the AgentRuleClassifier.

        Args:
            agent_name (str): Name identifier for the agent instance
        """
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_AgentRuleClassifier_{GLOBALS.rag_model_name}"
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    @property
    def rag_threshold(self):
        """
        Get the classification threshold value from globals.

        Returns:
            float: Minimum similarity score required for classification
        """
        return GLOBALS.classification_threshold  # with softmax. we fail if the score is lower.

    @property
    def softmax_temperature(self):
        """
        Get the softmax temperature value from globals.

        Returns:
            float: Temperature parameter for softmax calculation
        """
        return GLOBALS.classification_temperature

    def predict(self, query: str, samples_id, samples_embeddings) -> AgentMessage:
        """
        Classify a query string using the RAG system.

        This method embeds the query and finds the most similar rules in the database,
        applying softmax and threshold validation.

        Args:
            query (str): The text to be classified

        Returns:
            AgentMessage: A message object containing:
                - agent_name: Name of the classifier agent
                - agent_description: Description of the agent
                - agent_input: Original query
                - succeed: Whether classification was successful
                - agent_message: List of (rule_name, similarity_score) tuples
                - message_model: Same as agent_message
                - infer_time: Time taken for classification
        """
        start = time.time()

        # agent logic
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query)[0]
        rules_list = self.get_close_types_names(
            query_embedding=query_embeddings,
            samples_ids=samples_id,
            samples_embeddings=samples_embeddings,
            softmax=self.softmax,
            temperature=self.softmax_temperature
        )

        succeed = False
        if rules_list:
            closest_distance = rules_list[0][1]

            # Validate based on threshold and difference
            if closest_distance > self.rag_threshold:
                succeed = True
        else:
            rules_list = [(None, None)]

        response = rules_list

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=response,
            message_model=rules_list,
            infer_time=time.time() - start
        )
        return agent_message

    def get_rule_embeddings(self, rule_name: str, query_to_embed: str) -> tuple[bool, int | None, str, list[float]]:
        """
        Add a single rule to the classifier.

        Args:
            rule_name (str): Identifier for the rule
            query_to_embed (str): Text content to be embedded and stored as a rule

        Returns:
            tuple containing:
                - success (bool): Whether the rule was successfully added
                - index (int | None): Index of the rule if it already existed, None otherwise
                - rule_name (str): The name of the added rule
                - query_embeddings (list[float]): The embedded representation of the rule
        """
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query_to_embed)[0]
        return rule_name, query_embeddings

    def get_ruleS_embeddings(self, rules_names: list[str], queries_to_embed: list[str]) -> tuple[list[str], list[list[float]]]:
        """
        Add multiple rules to the classifier.

        WARNING: This method erases all existing rules before adding new ones!

        Args:
            rules_names (list[str]): List of rule identifiers
            queries_to_embed (list[str]): List of text contents to be embedded

        Returns:
            tuple containing:
                - rules_names (list[str]): List of added rule names
                - queries_embeddings (list[list[float]]): List of embedded representations
        """
        queries_embeddings: list[list[float]] = MODELS_MANAGER[self.model_nickname].infer(queries_to_embed)
        return rules_names, queries_embeddings

    def get_close_types_names(
            self,
            query_embedding: list[float],
            samples_ids = list[str],
            samples_embeddings = list[list[float]],
            *,
            softmax: bool = True,
            temperature: float = 0
    ):
        """
        Find the most similar samples to a query embedding.

        Computes cosine similarity between the query embedding and all stored embeddings,
        optionally applying softmax normalization with temperature scaling.

        Args:
            query_embedding (list[float]): The embedding vector to compare against stored samples
            softmax (bool, optional): Whether to apply softmax normalization to similarities. Defaults to True
            temperature (float, optional): Temperature parameter for softmax scaling. Defaults to 0
                Higher values make the probability distribution more uniform
                Lower values make it more concentrated on the highest similarities

        Returns:
            list[tuple[str, float]] | bool: If samples exist, returns list of tuples containing:
                - sample_id (str): The identifier of the sample
                - similarity (float): The similarity score (cosine similarity or softmax probability)
                If no samples exist, returns False

        Note:
            Results are sorted by similarity in descending order
        """
        rule_names_result = []

        if len(samples_ids) < 1:
            print("there's no samples")
            return False

        array_similarity = cosine_similarity([query_embedding], samples_embeddings)[0]
        if softmax:
            array_similarity = self.softmax_with_temperature(logits=array_similarity, temperature=temperature)
        indexes = np.argsort(array_similarity)[::-1]
        # adding the rules names and their score
        for i in range(len(samples_ids)):
            rule_names_result.append((samples_ids[indexes[i]], array_similarity[indexes[i]]))

        return rule_names_result

    @staticmethod
    def softmax_with_temperature(logits, temperature=1.0):
        """
        Apply softmax with temperature scaling to a vector of logits.

        The temperature parameter controls the "sharpness" of the probability distribution:
        - temperature > 1.0 makes the distribution more uniform
        - temperature < 1.0 makes the distribution more peaked

        Args:
            logits (numpy.ndarray): Input vector of similarity scores
            temperature (float, optional): Temperature scaling factor. Defaults to 1.0

        Returns:
            numpy.ndarray: Softmax probabilities with temperature scaling
        """
        logits = np.array(logits)
        scaled_logits = logits / max(temperature, 1e-8)
        exps = np.exp(scaled_logits - np.max(scaled_logits))  # Stability adjustment
        return exps / np.sum(exps)