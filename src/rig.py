from .rule_instance_generator import RuleInstanceGenerator
from Ollamamia.agents_manager import AgentsStore, AgentsManager
from .globals import GLOBALS
from .databases import DbRules, DbExamples, DbUnknowns
from .new_type import AddNewTypes
from .evaluation import evaluate_func
import pandas as pd

import concurrent.futures
import psutil
import time
from statistics import mean


class Rig:

    def __init__(self):

        self.db_rules = DbRules()
        self.db_examples = DbExamples()
        self.db_unknown = DbUnknowns()

        self.add_new_types = AddNewTypes()
        self.rule_instance_generator = RuleInstanceGenerator()

        self.agents_manager = AgentsManager()  # ollamamia -> where the agents are

        self.agents_manager[GLOBALS.summarization_agent] = AgentsStore.agent_summarization
        self.agents_manager[GLOBALS.rule_classifier_agent] = AgentsStore.agent_rule_classifier
        self.agents_manager[GLOBALS.examples_finder_agent] = AgentsStore.agent_examples_classifier
        self.agents_manager[GLOBALS.rule_instance_generator_agent] = AgentsStore.async_agent_generate_schema

        # add existing rules into agent
        rules_names = self.db_rules.df["rule_name"].tolist()
        embedded_rules = self.db_rules.df["embeddings"].tolist()
        self.agents_manager[GLOBALS.rule_classifier_agent].add_embedded_rules(rules_names, embedded_rules)

        # add existing examples into agent
        examples_names = self.db_examples.df["free_text"].tolist()
        embedded_examples = self.db_examples.df["embeddings"].tolist()
        self.agents_manager[GLOBALS.examples_finder_agent].add_embedded_examples(examples_names, embedded_examples)

    def get_rule_instance(self, free_text: str) -> dict:
        """
        Generates a rule instance from free text input.

        This method triggers agents to parse and classify input text, attempting to
        match it with existing rules and generate new instances.

        Args:
            free_text (str): The input text to be parsed and evaluated.
        Returns:
            dict: A dictionary containing the generated rule instance and related metadata.
            dict_keys(['query', 'message', 'is_error', 'agents_massages', 'total_infer_time', 'Uuid', 'dateTtime', 'rule_name', 'rule_instance_params', 'confidence', 'error_message', 'rule_instance'])
        """
        # init the agents flow (the data from old inference)
        self.agents_manager.new()

        response = self.rule_instance_generator.predict(self.agents_manager, self.db_rules, self.db_examples,
                                                        free_text=free_text)

        self.feedback(rig_response=response.copy())
        return response

    def get_rules_names(self) -> list:
        """
        Retrieves all rule names from the rule database.
        Returns:
            list: A list of rule names stored in the rules database.
        """
        return self.db_rules.df['rule_name'].tolist()

    def get_rule_details(self, rule_name: str) -> dict:
        """
        Fetches details of a specific rule.
        Args:
            rule_name (str): The name of the rule to retrieve.
        Returns:
            dict: A dictionary containing the rule's schema, description, and other metadata.
        """
        return self.db_rules.df[self.db_rules.df['rule_name'] == rule_name].to_dict(orient='records')[0]

    def set_rules(self, rule_types: list[dict] | None = None) -> bool:
        """
        Loads and embeds new rules into the rule database and agent.
        Args:
            rule_types (list[dict] | None): A list of dictionaries representing new rules.
        Returns:
            bool: True if the rules were successfully added and saved.
        """

        # get all the fields and the queries to embed
        rules_fields, chunks_to_embed = self.add_new_types.load(rule_types=rule_types)

        # agent embed and add everything to the agent data
        rules_names = [rule['rule_name'] for rule in rules_fields]
        rules_names, rules_embeddings = self.agents_manager[GLOBALS.rule_classifier_agent].add_ruleS(rules_names,
                                                                                                     chunks_to_embed)
        for i in range(len(rules_fields)):
            rules_fields[i]["embeddings"] = rules_embeddings[i]

        # add to the db for future loading
        self.db_rules.df = pd.DataFrame(rules_fields)
        self.db_rules.save_db()
        return True

    def add_rule(self, rule_type: dict | str) -> bool:
        """
        Adds a new rule to the database and updates agent embeddings.
        Args:
            rule_type (dict | str): A dictionary containing rule fields, or a string
                                    representing the rule.
        Returns:
            bool: True if the rule was successfully added.
        """

        # get all the fields and the queries to embed
        rule_fields, words_to_embed = self.add_new_types.add(rule_type=rule_type)

        # agent embed and add everything to the agent data
        success, index, rule_name, rule_embeddings = self.agents_manager[GLOBALS.rule_classifier_agent].add_rule(
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
            classification_threshold: float = GLOBALS.classification_threshold,
            classification_temperature: float = GLOBALS.classification_temperature,
            examples_rag_threshold: float = GLOBALS.examples_rag_threshold
    ) -> bool:
        GLOBALS.classification_threshold = classification_threshold
        GLOBALS.examples_rag_threshold = examples_rag_threshold
        GLOBALS.classification_temperature = classification_temperature
        return True

    def feedback(self, rig_response: dict, good: bool = None) -> bool:
        """
        Provides feedback on rig responses by updating the unknowns database or
        adding new examples.

        Args:
            rig_response (dict): The response generated by the system.
            good (bool | None): Indicates if the response was correct (True) or incorrect (False).

        Returns:
            bool: True if the feedback was processed successfully.
        """
        rig_response["good"] = good

        query_value = rig_response["query"]
        new_row = [str(rig_response.get(col, None)) for col in self.db_unknown.columns]

        query_mask = self.db_unknown.df["query"] == query_value  # Check if query already exists in database

        if query_mask.any():  # if already exist - update
            self.db_unknown.df.loc[query_mask] = new_row
        else:
            self.db_unknown.df.loc[len(self.db_unknown.df)] = new_row
        self.db_unknown.save_db()
        if good:
            example = dict(
                id=rig_response["query"],
                free_text=rig_response["query"],
                rule_name=rig_response["rule_name"],
                schema=self.db_rules.df.loc[self.db_rules.df["rule_name"] == rig_response["rule_name"], "schema"].iloc[
                    0],
                description=
                self.db_rules.df.loc[self.db_rules.df["rule_name"] == rig_response["rule_name"], "description"].iloc[0],
                rule_instance_params=rig_response["rule_instance_params"]
            )

            success, index, example_name, example_embeddings = self.agents_manager[
                GLOBALS.examples_finder_agent].add_example(
                example["free_text"])
            if success:
                example["embeddings"] = example_embeddings
                if not index:
                    self.db_examples.df = pd.concat([self.db_examples.df, pd.DataFrame([example])], ignore_index=True)
                if index:
                    self.db_examples.df.loc[index] = example
                self.db_examples.save_db()
        return True

    def evaluate(
            self,
            start_point=0,
            end_point=2,  # -1 - all the data
            jump=1,
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
            jump=jump,
            sleep_time_each_10=sleep_time_each_10_iter,
            batch_size=batch_size
        )

    def metadata(self, duration=60, interval=1) -> dict:
        """
        give basic data about the program and the resources usage
        :param duration:
        :param interval:
        :return:
        """

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(self.monitor_system_resources, duration, interval)
            system_resources = future.result()  # Get the result of the monitoring

        globals_data = vars(GLOBALS)
        agents_data = str(self.agents_manager)
        return dict(globals_data=globals_data, agents_data=agents_data, system_resources=system_resources)

    def monitor_system_resources(self, duration=60, interval=1):
        """
        Monitor system resources for a specified duration and return average usage.

        Args:
            duration (int): Total monitoring time in seconds (default: 60)
            interval (int): Sampling interval in seconds (default: 1)

        Returns:
            dict: Average CPU, RAM, and disk usage percentages
        """
        cpu_usage = []
        ram_usage = []
        disk_usage = []

        end_time = time.time() + duration

        while time.time() < end_time:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage.append(cpu_percent)

            # RAM usage
            memory = psutil.virtual_memory()
            ram_percent = memory.percent
            ram_usage.append(ram_percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_usage.append(disk_percent)

            time.sleep(interval)

        return {
            'average_cpu': round(mean(cpu_usage), 2),
            'average_ram': round(mean(ram_usage), 2),
            'average_disk': round(mean(disk_usage), 2),
            'samples_collected': len(cpu_usage)
        }
