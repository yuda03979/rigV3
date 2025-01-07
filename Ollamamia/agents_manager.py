
from Ollamamia.agents.classifiers.agent_rule_classifier import AgentRuleClassifier
from Ollamamia.agents.classifiers.agent_examples_classifier import AgentExamplesClassifier
from Ollamamia.agents.generators.agent_generate_schema import AgentGenerateSchema
from Ollamamia.agents.generators.async_agent_generate_scema import AsyncAgentGenerateSchema
from Ollamamia.agents.generators.agent_summarization import AgentSummarize
from Ollamamia.globals_dir.utils import AgentsFlow

import enum
import time


class AgentsStore(enum.Enum):
    agent_rule_classifier = AgentRuleClassifier
    agent_generate_schema = AgentGenerateSchema
    agent_examples_classifier = AgentExamplesClassifier
    async_agent_generate_schema = AsyncAgentGenerateSchema
    agent_summarization = AgentSummarize


class AgentsManager:

    def __init__(self):
        self.agents = {}
        self.agents_flow = None
        self.start = None

    def add_agents(self, agent_nickname: str, agent_name: AgentsStore):
        """
        maybe to add option to search agents with free text.
        :param agent_nickname:
        :param agent_name:
        :return:
        """
        if agent_nickname in self.agents.keys():
            # here need to unload the previous agent from the RAM
            print(f"agent_nickname {agent_nickname} already exist. overwriting...")

        self.agents[agent_nickname] = agent_name.value(agent_name=agent_nickname)

    def predict(self, model_nickname, query: dict | str):

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
        delete the data from previous covesations
        :return: None
        """
        del self.agents_flow
        self.agents_flow = None
        self.start = None

    def get_agents_flow(self):
        self.agents_flow.total_infer_time = time.time() - self.start
        return self.agents_flow

    def __setitem__(self, key: str, value: AgentsStore):
        self.add_agents(agent_nickname=key, agent_name=value)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.predict(item.start, item.stop)
        else:
            return self.agents.get(item)

    def cusbara(self):
        """
        opening the termianl for q&a
        :return:
        """
        pass
