import pydantic

from Ollamamia.src.agents_store import AgentsStore
from .globals import GLOBALS
from .databases import DbRules, DbExamples
from .new_type import AddNewTypes
import pandas as pd


class Rig:

    def __init__(self):

        self.db_rules = DbRules()
        self.db_examples = DbExamples()

        self.add_new_types = AddNewTypes()

        self.agents_store = AgentsStore()

        self.agents_store[GLOBALS.rule_classifier_agent] = "AgentRuleClassifier"
        self.agents_store[GLOBALS.examples_finder_agent] = "AgentExamplesClassifier"
        self.agents_store[GLOBALS.rule_instance_generator_agent] = "AgentGenerateSchema"

        # add existing rules into agent
        rules_names = self.db_rules.df["rule_name"].tolist()
        embedded_rules = self.db_rules.df["embeddings"].tolist()
        self.agents_store[GLOBALS.rule_classifier_agent].add_embedded_rules(rules_names, embedded_rules)

        # add existing examples into agent

    def get_rule_instance(self, free_text: str) -> dict:
        # init the agents flow (the data from old inference)
        self.agents_store.new()

        #######
        # classify the rule name
        agent_message = self.agents_store[GLOBALS.rule_classifier_agent:free_text]

        if not agent_message.succeed:
            return
        rule_name = agent_message.agent_message

        #######
        # with the rule_name get from the db_rules the schema and description
        schema = self.db_rules.df.loc[self.db_rules.df["rule_name"] == rule_name, "schema"].iloc[0]
        description = self.db_rules.df.loc[self.db_rules.df["rule_name"] == rule_name, "description"].iloc[0]

        #######
        # get examples

        # agent_message = self.agents_store[GLOBALS.examples_finder_agent:free_text]
        # if not agent_message.succeed:
        #     example1 = None
        #     example2 = None
        # example1 = agent_message.agent_message[0]
        # example2 = agent_message.agent_message[1]

        example1 = None
        example2 = None
        #######
        # generate the rule instance

        agent_message = self.agents_store[GLOBALS.rule_instance_generator_agent:dict(
            query=free_text,
            schema=str(schema),
            rule_name=rule_name,
            example1=str(example1),
            example2=str(example2),
            description=str(description)
        )]

        agents_flow: pydantic.BaseModel = self.agents_store.get_agents_flow()
        return agents_flow.model_dump()

    def get_rule_types_names(self) -> list:
        return self.db_rules.df['rule_name'].tolist()

    def get_rule_type_details(self, rule_name: str) -> dict:
        return self.db_rules.df[self.db_rules.df['rule_name'] == rule_name].to_dict(orient='records')[0]

    def set_rule_types(self, rule_types: list[dict] = None) -> None:
        # get all the fields and the queries to embed
        rules_fields, chunks_to_embed = self.add_new_types.load(rule_types=rule_types)

        # agent embed and add everything to the agent data
        rules_names = [rule['rule_name'] for rule in rules_fields]
        rules_names, rules_embeddings = self.agents_store[GLOBALS.rule_classifier_agent].add_ruleS(rules_names,
                                                                                                   chunks_to_embed)
        for i in range(len(rules_fields)):
            rules_fields[i]["embeddings"] = rules_embeddings[i]

        # add to the db for future loading
        self.db_rules.df = pd.DataFrame(rules_fields)
        self.db_rules.save_db()

    def add_rule_type(self, rule_type: dict = None) -> None:
        # get all the fields and the queries to embed
        rule_fields, words_to_embed = self.add_new_types.add(rule_type=rule_type)

        # agent embed and add everything to the agent data
        success, index, rule_name, rule_embeddings = self.agents_store[GLOBALS.rule_classifier_agent].add_rule(
            rule_fields["rule_name"], words_to_embed)
        rule_fields["embeddings"] = rule_embeddings

        # add to the db for future loading
        if success:
            if index:
                self.db_rules.df.loc[index] = rule_fields
            else:
                self.db_rules.append(rule_fields)
            self.db_rules.save_db()

    def tweak_parameters(
            self,
            rag_threshold: float,
    ) -> None:
        pass

    def feedback(self, rig_response: dict, good: bool) -> dict:
        pass

    def evaluate(
            self,
            start_point=0,
            end_point=2,  # -1 - all the data
            sleep_time_each_10_iter=30,
            batch_size=250
    ):
        pass