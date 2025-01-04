import time

from agents.classifiers.agent_rule_classifier import AgentRuleClassifier
from agents.generators.agent_generate_schema import AgentGenerateSchema
from agents.classifiers.agent_examples_classifier import AgentExamplesClassifier
from globals_dir.utils import AgentsFlow

class AgentsStore:

    def __init__(self):
        self.agents = {}
        self.agents_available = {
            "AgentRuleClassifier": AgentRuleClassifier,
            "AgentGenerateSchema": AgentGenerateSchema,
            "AgentExamplesClassifier": AgentExamplesClassifier
        }

        self.store = ""  # enum for available models
        self.agents_flow = None
        self.start = None


    def add_agents(self, agent_nickname: str, agent_name: str):
        """
        maybe to add option to search agents with free text.
        :param agent_nickname:
        :param agent_name:
        :return:
        """
        if not self.agents_available.get(agent_name):
            print(f"your agent {agent_name} do not exist. choose one of those: {list(self.agents_available.keys())}")
            return

        if agent_nickname in self.agents.keys():
            # here need to unload the previous agent from the RAM
            print(f"agent_nickname {agent_nickname} already exist. overwriting...")
        self.agents[agent_nickname] = self.agents_available[agent_name](agent_name=agent_nickname)

    def predict(self, model_nickname, query: dict | str):
        if not model_nickname in self.models.keys():
            handle_errors(e=f"model_nickname: {model_nickname} do not exist. \nexisting nicknames: {list(self.agents.keys())}")
        if not self.agents_flow:
            self.agents_flow = AgentsFlow(query=query)
            self.start = time.time()
        agent_message = self.agents[model_nickname].predict(query)
        if agent_message.succeed == False:
            self.agents_flow.is_error = True
        self.agents_flow.append(agent_message)
        return self.agents_message

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

    def __setitem__(self, key: str, value: str):
        self.add_agents(agent_nickname=key, agent_name=value)

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.predict(item.start, item.stop)
        else:
            return self.agents.get(item)

    def cusbara(self):
        """
        opening the termianl for q&a
        :return:
        """
        pass


