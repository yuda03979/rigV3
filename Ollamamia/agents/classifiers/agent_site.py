from Ollamamia.globals_dir.models_manager import MODELS_MANAGER
from Ollamamia.globals_dir.utils import AgentMessage
from Ollamamia.agents.logic.basic_rag import BasicRag
from src.globals import GLOBALS
import time


class AgentSiteClassifier:
    """
    A site-based classifier agent using RAG (Retrieval-Augmented Generation) for text classification.

    This agent is designed for small to medium-sized databases and uses embeddings to classify
    text inputs according to predefined sites. It leverages the Ollama framework for embedding
    generation and supports both single and batch site additions.

    Attributes:
        description (str): Description of the RAG implementation
        model_nickname (str): Unique identifier for the model instance
        engine (str): The engine used for embeddings ("ollama")
        model_name (str): Name of the model from GLOBALS
        task (str): Type of task the model performs ("embed")
        max_sites (int): Maximum number of sites that can be stored (default: 100,000)
        prefix (str): Prefix added to improve RAG accuracy
        softmax (bool): Whether to apply softmax to similarity scores
    """

    description = """site rag implemented for elta, suitable for small - medium size db. """
    engine = "ollama"
    model_name = GLOBALS.rag_model_name  # "snowflake-arctic-embed:137m"

    task = "embed"
    max_sites: int = 100_000
    prefix: str = "classification:\n"
    softmax: bool = True

    def __init__(self, agent_name: str):
        """
        Initialize the AgentRuleClassifier.

        Args:
            agent_name (str): Name identifier for the agent instance
        """
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_AgentSiteClassifier_{GLOBALS.rag_model_name}"
        self.basic_rag = BasicRag(max_sites=self.max_sites)
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    @property
    def site_rag_threshold(self):
        """
        Get the classification threshold value from globals.

        Returns:
            float: Minimum similarity score required for classification
        """
        return GLOBALS.site_rag_threshold  # with softmax. we fail if the score is lower.

    @property
    def site_temperature(self):
        """
        Get the softmax temperature value from globals.

        Returns:
            float: Temperature parameter for softmax calculation
        """
        return GLOBALS.site_temperature

    def predict(self, query: str) -> AgentMessage:
        """
        Classify a query string using the RAG system.

        This method embeds the query and finds the most similar sites in the database,
        applying softmax and threshold validation.

        Args:
            query (str): The text to be classified

        Returns:
            AgentMessage: A message object containing:
                - agent_name: Name of the classifier agent
                - agent_description: Description of the agent
                - agent_input: Original query
                - succeed: Whether classification was successful
                - agent_message: List of (site_name, similarity_score) tuples
                - message_model: Same as agent_message
                - infer_time: Time taken for classification
        """
        start = time.time()

        # agent logic
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query)[0]
        sites_list = self.basic_rag.get_close_types_names(
            query_embedding=query_embeddings,
            softmax=self.softmax,
            temperature=self.site_temperature
        )

        succeed = False
        if sites_list:
            closest_distance = sites_list[0][1]

            # Validate based on threshold and difference
            if closest_distance > self.site_rag_threshold:
                succeed = True
        else:
            sites_list = [(None, None)]

        response = sites_list

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=response,
            message_model=sites_list,
            infer_time=time.time() - start
        )
        return agent_message

    def add_site(self, site: str) -> tuple[bool, int | None, str, list[float]]:
        """
        Add a single site to the classifier.

        Args:
            site_name (str): Identifier for the site
            query_to_embed (str): Text content to be embedded and stored as a site

        Returns:
            tuple containing:
                - success (bool): Whether the site was successfully added
                - index (int | None): Index of the site if it already existed, None otherwise
                - site_name (str): The name of the added site
                - query_embeddings (list[float]): The embedded representation of the site
        """
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(site)[0]
        success, index = self.basic_rag.add_sample(sample_id=site, sample_embeddings=query_embeddings)
        return success, index, site, query_embeddings

    def add_siteS(self, sites_names: list[str]) -> tuple[list[str], list[list[float]]]:
        """
        Add multiple sites to the classifier.

        WARNING: This method erases all existing sites before adding new ones!

        Args:
            sites_names (list[str]): List of site identifiers
            queries_to_embed (list[str]): List of text contents to be embedded

        Returns:
            tuple containing:
                - sites_names (list[str]): List of added site names
                - queries_embeddings (list[list[float]]): List of embedded representations
        """
        queries_embeddings: list[list[float]] = MODELS_MANAGER[self.model_nickname].infer(sites_names)
        self.basic_rag.add_samples(samples_ids=sites_names, sample_embeddings=queries_embeddings)
        return sites_names, queries_embeddings

    def add_embedded_sites(self, sites: list[str], embedded_sites: list[list[float]]) -> None:
        """
        Add pre-embedded sites to the classifier.

        Args:
            sites (list[str]): List of site identifiers
            embedded_sites (list[list[float]]): List of pre-computed embeddings

        Note:
            This method assumes the embeddings were generated using the same model
            and configuration as the current classifier.
        """
        self.basic_rag.add_samples(samples_ids=sites, sample_embeddings=embedded_sites)

    def remove_site(self, site_name) -> bool:
        return self.basic_rag.remove_sample(sample_id=site_name)
