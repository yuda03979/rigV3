from ...globals_dir.models_manager import MODELS_MANAGER
from ...globals_dir.utils import AgentMessage
from ..logic.basic_rag import BasicRag
import time


class AgentRuleClassifier:
    description = """rag implemented for elta, suitable for small - medium size db. """

    model_nickname = str(MODELS_MANAGER.get_num_models())
    engine = "ollama"
    model_name = "snowflake-arctic-embed:137m"
    task = "embed"

    rag_threshold: float = 0.5  # if the highest similarity is under this - we failed.

    max_rules: int = 100_000
    prefix: str = "classification: \n"
    softmax: bool = True
    softmax_temperature: float = 0

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_{self.model_nickname}"
        self.basic_rag = BasicRag(max_rules=self.max_rules)
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

        succeed = False
        if rules_list:
            closest_distance = rules_list[0][1]

            # Validate based on threshold and difference
            if closest_distance > self.rag_threshold:
                succeed = True

        ####################

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=rules_list[0][0],
            message_model=rules_list,
            infer_time=time.time() - start
        )
        return agent_message

    def add_rule(self, rule_name: str, query_to_embed: str):
        """

        :param rule_name:
        :param query_to_embed:
        :return:
            success: bool,
            index: int - if already exist,
            rule_name: str,
            query_embeddings: list[float]
        """
        query_to_embed = self.prefix + query_to_embed
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query_to_embed)[0]
        success, index = self.basic_rag.add_sample(sample_id=rule_name, sample_embeddings=query_embeddings)
        return success, index, rule_name, query_embeddings

    def add_ruleS(self, rules_names: list[str], queries_to_embed: list[str]) -> tuple[list[str], list[list[float]]]:
        """
        ERASING EXISTING SAMPLES!!
        :param rules_names:
        :param queries_embeddings:
        :return:
        """
        queries_embeddings: list[list[float]] = MODELS_MANAGER[self.model_nickname].infer(queries_to_embed)
        self.basic_rag.add_samples(samples_ids=rules_names, sample_embeddings=queries_embeddings)
        return rules_names, queries_embeddings

    def add_embedded_rules(self, rules: list[str], embedded_rules: list[list[float]]) -> None:
        self.basic_rag.add_samples(samples_ids=rules, sample_embeddings=embedded_rules)
