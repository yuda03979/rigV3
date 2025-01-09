from Ollamamia.agents.classifiers.agent_rule_classifier import AgentRuleClassifier
from Ollamamia.agents.classifiers.agent_examples_classifier import AgentExamplesClassifier
from Ollamamia.agents.generators.agent_generate_schema import AgentGenerateSchema
from Ollamamia.agents.generators.async_agent_generate_scema import AsyncAgentGenerateSchema
from Ollamamia.agents.generators.agent_summarization import AgentSummarize
from Ollamamia.globals_dir.utils import AgentsFlow

import enum
import time


class AgentsStore(enum.Enum):
    """
    Enumeration of available agent types in the system.

    This enum maps agent nicknames to their corresponding implementation classes,
    making it easy to reference and instantiate different types of agents.

    Available Agents:
        - agent_rule_classifier: For classifying rules
        - agent_generate_schema: For generating schemas
        - agent_examples_classifier: For classifying examples
        - async_agent_generate_schema: Async version of schema generation
        - agent_summarization: For text summarization
    """
    agent_rule_classifier = AgentRuleClassifier
    agent_generate_schema = AgentGenerateSchema
    agent_examples_classifier = AgentExamplesClassifier
    async_agent_generate_schema = AsyncAgentGenerateSchema
    agent_summarization = AgentSummarize


class AgentsManager:
    """
    Manages multiple AI agents, handling their lifecycle and interactions.

    This class provides a centralized way to create, manage, and interact with
    multiple AI agents. It tracks agent states, manages conversations, and
    provides convenient access patterns including dictionary-style access.

    Attributes:
        agents (dict): Dictionary mapping agent nicknames to agent instances
        agents_flow (AgentsFlow): Tracks the flow of conversation between agents
        start (float): Timestamp when the current conversation started

    Examples:
        manager = AgentsManager()
        manager["classifier"] = AgentsStore.agent_rule_classifier
        response = manager["classifier", "classify this text"]
        print(manager)  # Shows summary of loaded agents
    """

    def __init__(self):
        """
        Initializes an empty agents manager.
        """
        self.agents = {}
        self.agents_flow = None
        self.start = None

    def add_agents(self, agent_nickname: str, agent_name: AgentsStore):
        """
        Adds a new agent to the manager or replaces an existing one.

        Args:
            agent_nickname (str): Nickname to reference the agent by
            agent_name (AgentsStore): Type of agent to create from AgentsStore enum

        Note:
            If an agent with the same nickname exists, it will be overwritten.
            Future implementation may include cleanup of previous agent resources.
        """
        if agent_nickname in self.agents.keys():
            # here need to unload the previous agent from the RAM
            print(f"agent_nickname {agent_nickname} already exist. overwriting...")

        self.agents[agent_nickname] = agent_name.value(agent_name=agent_nickname)

    def predict(self, model_nickname, query: dict | str):
        """
        Makes a prediction using the specified agent.

        Args:
            model_nickname: Nickname of the agent to use
            query (dict | str): Input query for the agent

        Returns:
            agent_message: Response from the agent containing prediction results

        Note:
            Automatically initializes a new conversation flow if none exists.
            Updates the conversation flow with the agent's response.
        """
        if not model_nickname in self.agents.keys():
            print(f"model_nickname: {model_nickname} do not exist. \nexisting nicknames: {list(self.agents.keys())}")

        if not self.agents_flow:
            self.agents_flow = AgentsFlow(query=query)
            self.start = time.time()

        agent_message = self.agents[model_nickname].predict(query)

        if not agent_message.succeed:
            self.agents_flow.is_error = True

        self.agents_flow.append(agent_message)

        return agent_message

    def new(self):
        """
        Resets the conversation state, starting a fresh conversation.

        Deletes the existing agents_flow and resets timing information.
        """
        del self.agents_flow
        self.agents_flow = None
        self.start = None

    def get_agents_flow(self):
        """
        Returns the current conversation flow with timing information.

        Returns:
            AgentsFlow: Object containing the conversation history and metadata
        """
        self.agents_flow.total_infer_time = time.time() - self.start
        return self.agents_flow

    def __setitem__(self, key: str, value: AgentsStore):
        """
        Enables dictionary-style agent addition: manager["nickname"] = AgentsStore.agent_type
        """
        self.add_agents(agent_nickname=key, agent_name=value)

    def __getitem__(self, item):
        """
        Enables dictionary-style agent access and prediction:
        - manager["nickname"] returns the agent
        - manager["nickname", query] makes a prediction
        """
        if isinstance(item, slice):
            return self.predict(item.start, item.stop)
        else:
            return self.agents.get(item)

    def __repr__(self):
        """
        Provides a detailed string representation of the manager's state.

        Returns:
            str: Dictionary containing:
                - Number of loaded agents
                - List of agent nicknames
                - List of loaded model names
                - List of model nicknames
        """
        num_agents = len(self.agents)
        agents_nicknames = list(self.agents.keys())
        models_loaded = [agent.model_name for agent in self.agents.values()]
        models_nicknames = [agent.model_nickname for agent in self.agents.values()]
        return str(dict(
            num_agents=num_agents,
            agents_nicknames=agents_nicknames,
            models_loaded=models_loaded,
            models_nicknames=models_nicknames
        ))