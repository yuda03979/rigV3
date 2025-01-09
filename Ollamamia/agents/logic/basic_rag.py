from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BasicRag:
    """
    A basic Retrieval-Augmented Generation (RAG) implementation that stores and retrieves
    embeddings using cosine similarity.

    This class provides functionality to store sample embeddings with their corresponding IDs
    and retrieve the most similar samples based on a query embedding. It supports features
    like maximum capacity control and similarity scoring with optional softmax normalization.

    Attributes:
        max_rules (int): Maximum number of samples that can be stored
        samples_id (list[str]): List of sample identifiers
        samples_embeddings (list[list[float]]): List of embedding vectors corresponding to samples
    """

    def __init__(self, max_rules: int):
        """
        Initialize the BasicRag instance.

        Args:
            max_rules (int): Maximum number of samples that can be stored in the database
        """
        self.max_rules = max_rules
        self.samples_id = []
        self.samples_embeddings = []

    def add_sample(self, sample_id: str, sample_embeddings: list[float]) -> tuple[bool, None | int]:
        """
        Add a single sample with its embedding to the database.

        If the sample_id already exists, it updates the existing entry.
        If the database is full (reached max_rules), the addition fails.

        Args:
            sample_id (str): Unique identifier for the sample
            sample_embeddings (list[float]): Embedding vector for the sample

        Returns:
            tuple[bool, None | int]: A tuple containing:
                - success (bool): True if addition/update was successful, False otherwise
                - index (None | int): Index of the updated sample if it existed, None otherwise
        """
        success = True
        index = None

        # if sample_id already exist its replacing the oldest with the newest
        if sample_id in self.samples_id:
            index = self.samples_id.index(sample_id)
            self.samples_id[index] = sample_id
            self.samples_embeddings[index] = sample_embeddings
            return success, index

        # if you add more than self.max_rules, it will not add more. (its like 100_000 its kind of 35MB)
        elif len(self.samples_id) >= self.max_rules:
            print(f"cant add more rules! your db > {self.max_rules}")
            success = False
            return success, index

        # adding the rule.
        else:
            self.samples_id.append(sample_id)
            self.samples_embeddings.append(sample_embeddings)
            return success, index

    def add_samples(self, samples_ids: list[str], sample_embeddings: list[list[float]]):
        """
        Replace all existing samples with a new batch of samples.

        WARNING: This method completely erases existing samples!

        Args:
            samples_ids (list[str]): List of sample identifiers
            sample_embeddings (list[list[float]]): List of embedding vectors corresponding to samples_ids

        Note:
            The lengths of samples_ids and sample_embeddings must match
        """
        self.samples_id = samples_ids
        self.samples_embeddings = sample_embeddings

    def get_close_types_names(
            self,
            query_embedding: list[float],
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

        if len(self.samples_id) < 1:
            print("there's no samples")
            return False

        array_similarity = cosine_similarity([query_embedding], self.samples_embeddings)[0]
        if softmax:
            array_similarity = self.softmax_with_temperature(logits=array_similarity, temperature=temperature)
        indexes = np.argsort(array_similarity)[::-1]
        # adding the rules names and their score
        for i in range(len(self.samples_id)):
            rule_names_result.append((self.samples_id[indexes[i]], array_similarity[indexes[i]]))

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