from typing import Literal

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BasicRag:

    def __init__(self, max_rules: int):
        self.max_rules = max_rules
        self.samples_id = []
        self.samples_embeddings = []

    def add_sample(self, sample_id: str, sample_embeddings: list[float]) -> tuple[bool, None | int]:

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
        ERASING EXISTING SAMPLES!!
        :param samples_ids:
        :param sample_embeddings:
        :return:
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
        Classifies a free-form sample to the closest rule_type.
        :param query_embedding:
        :param temperature:
        :param softmax:
        :return: list in shape [(type_name, similarity), ...] with length of len_response.
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
        logits = np.array(logits)
        scaled_logits = logits / max(temperature, 1e-8)
        exps = np.exp(scaled_logits - np.max(scaled_logits))  # Stability adjustment
        return exps / np.sum(exps)
