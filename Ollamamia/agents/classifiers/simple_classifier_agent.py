from typing import Tuple, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Ollamamia.globals_dir.models_manager import MODELS_MANAGER
from Ollamamia.globals_dir.utils import AgentMessage
from src.globals import GLOBALS
import time


class AgentSimpleClassifier:
    """
    A sample-based classifier agent using RAG (Retrieval-Augmented Generation) for text classification.

    This agent is designed for small to medium-sized databases and uses embeddings to classify
    text inputs according to predefined samples. It leverages the Ollama framework for embedding
    generation and supports both single and batch sample additions.

    Attributes:
        description (str): Description of the RAG implementation
        model_nickname (str): Unique identifier for the model instance
        engine (str): The engine used for embeddings ("ollama")
        model_name (str): Name of the model from GLOBALS
        task (str): Type of task the model performs ("embed")
        prefix (str): Prefix added to improve RAG accuracy
    """

    description = """basic rag implemented for elta, suitable for small - medium size db"""
    engine = "ollama"
    model_name = GLOBALS.rag_model_name  # "snowflake-arctic-embed:137m"

    task = "embed"
    prefix: str = "classification:\n"

    def __init__(self, agent_name: str):
        """
        Initialize the AgentRuleClassifier.

        Args:
            agent_name (str): Name identifier for the agent instance
        """
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_{GLOBALS.rag_model_name}"
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    @property
    def softmax_temperature(self):
        """
        Get the softmax temperature value from globals.

        Returns:
            float: Temperature parameter for softmax calculation
        """
        return GLOBALS.rag_temperature

    def predict(self, query: str, samples_ids, samples_embeddings, softmax=True) -> AgentMessage:
        """
        Classify a query string using the RAG system.

        This method embeds the query and finds the most similar samples in the database,
        applying softmax and threshold validation.

        Args:
            query (str): The text to be classified

        Returns:
            AgentMessage: A message object containing:
                - agent_name: Name of the classifier agent
                - agent_description: Description of the agent
                - agent_input: Original query
                - succeed: Whether classification was successful
                - agent_message: List of (sample_name, similarity_score) tuples
                - message_model: Same as agent_message
                - infer_time: Time taken for classification
        """
        start = time.time()

        # agent logic
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query)[0]
        samples_list = self.get_similarity(
            query_embedding=query_embeddings,
            samples_ids=samples_ids,
            samples_embeddings=samples_embeddings,
            softmax=softmax,
            temperature=self.softmax_temperature
        )

        succeed = False
        if samples_list:
            samples_list = samples_list[:10]
        else:
            samples_list = [(None, None)]

        response = samples_list[:2]

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=response,
            message_model=samples_list,
            infer_time=time.time() - start
        )
        return agent_message

    def get_sample_embeddings(self, sample_name: str, query_to_embed: str) -> tuple[str, list[float]]:
        """
        Add a single sample to the classifier.

        Args:
            sample_name (str): Identifier for the sample
            query_to_embed (str): Text content to be embedded and stored as a sample

        Returns:
            tuple containing:
                - success (bool): Whether the sample was successfully added
                - index (int | None): Index of the sample if it already existed, None otherwise
                - sample_name (str): The name of the added sample
                - query_embeddings (list[float]): The embedded representation of the sample
        """
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query_to_embed)[0]
        return sample_name, query_embeddings

    def get_sampleS_embeddings(self, samples_names: list[str], queries_to_embed: list[str]) -> tuple[
        list[str], list[list[float]]]:
        """
        Add multiple samples to the classifier.

        WARNING: This method erases all existing samples before adding new ones!

        Args:
            samples_names (list[str]): List of sample identifiers
            queries_to_embed (list[str]): List of text contents to be embedded

        Returns:
            tuple containing:
                - samples_names (list[str]): List of added sample names
                - queries_embeddings (list[list[float]]): List of embedded representations
        """
        queries_embeddings: list[list[float]] = MODELS_MANAGER[self.model_nickname].infer(queries_to_embed)
        return samples_names, queries_embeddings

    def get_similarity(
            self,
            query_embedding: list[float],
            samples_ids=list[str],
            samples_embeddings=list[list[float]],
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
        result = []

        if len(samples_ids) < 1:
            print("there's no samples")
            return False

        array_similarity = cosine_similarity([query_embedding], samples_embeddings)[0]
        if softmax:
            array_similarity = self.softmax_with_temperature(logits=array_similarity, temperature=temperature)
        indexes = np.argsort(array_similarity)[::-1]
        # adding the samples names and their score
        for i in range(len(samples_ids)):
            result.append((samples_ids[indexes[i]], array_similarity[indexes[i]]))
        return result

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
