from .rule_instance_generator import RuleInstanceGenerator
from Ollamamia.src.agents_store import AgentsStore
from .globals import GLOBALS
from .databases import DbRules, DbExamples
from .new_type import AddNewTypes
from .evaluation import evaluate_func
import pandas as pd


class Rig:

    def __init__(self):

        self.db_rules = DbRules()
        self.db_examples = DbExamples()

        self.add_new_types = AddNewTypes()
        self.rule_instance_generator = RuleInstanceGenerator()

        self.agents_store = AgentsStore()  # ollamamia -> where the agents are

        self.agents_store[GLOBALS.rule_classifier_agent] = "AgentRuleClassifier"
        self.agents_store[GLOBALS.examples_finder_agent] = "AgentExamplesClassifier"
        self.agents_store[GLOBALS.rule_instance_generator_agent] = "AgentGenerateSchema"

        # add existing rules into agent
        rules_names = self.db_rules.df["rule_name"].tolist()
        embedded_rules = self.db_rules.df["embeddings"].tolist()
        self.agents_store[GLOBALS.rule_classifier_agent].add_embedded_rules(rules_names, embedded_rules)

        # add existing examples into agent
        examples_names = self.db_examples.df["free_text"].tolist()
        embedded_examples = self.db_examples.df["embeddings"].tolist()
        self.agents_store[GLOBALS.examples_finder_agent].add_embedded_examples(examples_names, embedded_examples)

    def get_rule_instance(self, free_text: str) -> dict:
        # init the agents flow (the data from old inference)
        self.agents_store.new()

        response = self.rule_instance_generator.predict(self.agents_store, self.db_rules, self.db_examples, free_text=free_text)

        return response

    def get_rules_names(self) -> list:
        return self.db_rules.df['rule_name'].tolist()

    def get_rule_details(self, rule_name: str) -> dict:
        return self.db_rules.df[self.db_rules.df['rule_name'] == rule_name].to_dict(orient='records')[0]

    def set_rules(self, rule_types: list[dict] = None) -> bool:
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
        return True

    def add_rule(self, rule_type: dict = None) -> bool:
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
                self.db_rules.df.loc[len(self.db_rules.df)] = rule_fields
            self.db_rules.save_db()

        return True

    def tweak_parameters(
            self,
            rag_threshold: float,
            examples_rag_threshold: float
    ) -> bool:
        GLOBALS.rag_threshold = rag_threshold
        GLOBALS.examples_rag_threshold = examples_rag_threshold
        return True

    def feedback(self, rig_response: dict, good: bool) -> bool:
        example = dict(
            id=rig_response["query"],
            free_text=rig_response["query"],
            rule_name=rig_response["rule_name"],
            schema=self.db_rules.df.loc[self.db_rules.df["rule_name"] == rig_response["rule_name"], "schema"].iloc[0],
            description=self.db_rules.df.loc[self.db_rules.df["rule_name"] == rig_response["rule_name"], "description"].iloc[0],
            rule_instance_params=rig_response["rule_instance_params"]
        )

        if good:
            success, index, example_name, example_embeddings = self.agents_store[GLOBALS.examples_finder_agent].add_example(
                example["free_text"])
            if success:
                example["embeddings"] = example_embeddings
                if not index:
                    self.db_examples.df = pd.concat([self.db_examples.df, pd.DataFrame([example])], ignore_index=True)
                if index:
                    self.db_examples.df.loc[index] = example
                self.db_examples.save_db()
        else:
            pass

        return True

    def evaluate(
            self,
            start_point=0,
            end_point=2,  # -1 - all the data
            sleep_time_each_10_iter=30,
            batch_size=250
    ):
        # to do! return th actual scores and not the last one
        return evaluate_func(
            self,
            data_file_path=GLOBALS.evaluation_data_path,
            output_directory=GLOBALS.evaluation_output_dir,
            start_point=start_point,
            end_point=end_point,  # -1 all the data (almost...)
            sleep_time_each_10=sleep_time_each_10_iter,
            batch_size=batch_size
        )
