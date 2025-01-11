import ollama

from .rule_instance_generator import RuleInstanceGenerator
from Ollamamia.agents_manager import AgentsStore, AgentsManager
from .globals import GLOBALS
from .databases import DbRules, DbExamples, DbUnknowns, DbSites
from .new_type import AddNewTypes
from .evaluation import evaluate_func
from .metadata import metadata
import pandas as pd
import time


class Rig:

    def __init__(self):

        self.db_rules = DbRules()
        self.db_examples = DbExamples()
        self.db_unknown = DbUnknowns()
        self.db_sites = DbSites()

        self.add_new_types = AddNewTypes()
        self.rule_instance_generator = RuleInstanceGenerator()

        self.agents_manager = AgentsManager()  # ollamamia -> where the agents are

        # init the agents

        self.agents_manager[GLOBALS.summarization_agent] = AgentsStore.agent_summarization
        self.agents_manager[GLOBALS.classifier_agent] = AgentsStore.agent_simple_classifier

        if GLOBALS.run_async_models:
            self.agents_manager[GLOBALS.rule_instance_generator_agent] = AgentsStore.async_agent_generate_schema
        else:
            self.agents_manager[GLOBALS.rule_instance_generator_agent] = AgentsStore.agent_generate_schema

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
        start = time.time()
        self.agents_manager.new()

        response = self.rule_instance_generator.predict(
            self.agents_manager,
            self.db_rules,
            self.db_examples,
            free_text=free_text
        )
        response['total_infer_time'] = time.time() - start

        if response.get('confidence') == 1:
            self.feedback(rig_response=response.copy(), good=True)
        else:
            self.feedback(rig_response=response.copy())

        # if site or Site exist, get the closest site using agent, and get the site_id from the db.
        def get_site_id(site_field='site'):
            # if there's no error and site exist
            if not response["is_error"] and response["rule_instance"]["params"].get(site_field) not in [None, 'null']:
                # get similar site form agent
                agent_message = self.agents_manager[
                    GLOBALS.classifier_agent].predict(
                    query=str(response["rule_instance"]["params"][site_field]),
                    samples_ids=self.db_sites.df['site'].tolist(),
                    samples_embeddings=self.db_sites.df['embeddings'].tolist()
                )
                site_value = agent_message.agent_message[0][0]
                if site_value is None or agent_message.agent_message[0][1] < GLOBALS.site_rag_threshold:
                    response["rule_instance"]["params"][site_field] = site_value[0][0]
                    response['is_error'] = True
                    response["error_message"] = response["error_message"] + (f' closest site didnt succeed the '
                                                                             f'threshold: {agent_message.agent_message}'
                                                                             )
                # if agent recognizes it:
                else:
                    site = self.db_sites.df[self.db_sites.df[site_field] == site_value]['site_id'].iloc[0]
                    response["rule_instance"]["params"][site_field] = site
                    return True
            return False

        if not get_site_id('site'):
            get_site_id('Site')
        return response

    def tweak_parameters(
            self,
            rag_temperature: float = GLOBALS.rag_temperature,
            classification_threshold: float = GLOBALS.classification_threshold,
            site_rag_threshold: float = GLOBALS.site_rag_threshold,
            add_example_rag_threshold: float = GLOBALS.add_example_rag_threshold,
    ) -> bool:
        GLOBALS.rag_temperature = rag_temperature
        GLOBALS.classification_threshold = classification_threshold
        GLOBALS.site_rag_threshold = site_rag_threshold
        GLOBALS.add_example_rag_threshold = add_example_rag_threshold
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
                schema=self.db_rules.df.loc[self.db_rules.df["rule_name"] == rig_response["rule_name"], "schema"].iloc[0],
                description=self.db_rules.df.loc[self.db_rules.df["rule_name"] == rig_response["rule_name"], "description"].iloc[0],
                rule_instance_params=rig_response["rule_instance_params"],
                embeddings=self.agents_manager[GLOBALS.classifier_agent].get_sample_embeddings(
                    sample_name=rig_response['query'],
                    query_to_embed=rig_response['query']
                )[1],
                usage="0"
            )
            # check if it too close to other examples:
            examples = self.agents_manager[GLOBALS.classifier_agent].predict(example['free_text'],
                                                                             self.db_examples.df['free_text'].tolist(),
                                                                             self.db_examples.df['embeddings'].tolist()
                                                                             ).agent_message
            if examples[0][0] is None or examples[0][1] < GLOBALS.add_example_rag_threshold:
                print('didnt add the example')
                return True
            try:
                index = self.db_examples.df['rule_name'].tolist().index(example['free_text'])
                self.db_examples.df.loc[index] = example
            except ValueError:
                self.db_examples.remove_unused()
                self.db_examples.df.loc[len(self.db_rules.df)] = example
            self.db_examples.save_db()
        return True

    def evaluate(
            self,
            start_point=0,
            end_point=2,  # -1 - all the data
            jump=1,
            sleep_time_each_10_iter=30,
            batch_size=250,
            set_eval_rules=True  # deleting existing rules!!! and loading the directory
    ):
        if set_eval_rules:
            self.set_rules(_eval=False)
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

    def metadata(self) -> dict:
        """
        give basic data about the program and the resources usage
        :return: dict
        """
        globals_data = {str(k): str(v) for k, v in GLOBALS.__class__.__dict__.items()}
        response = metadata(self)
        response["globals_data"] = globals_data
        response["ollama_models"] = dict(existing_models=ollama.list(), loaded_models=ollama.ps())
        return response

    def restart(self, db_rules: bool = False, db_examples: bool = False, db_sites: bool = False,
                _db_unknown: bool = False):
        """deleting the db's"""
        if db_rules:
            self.db_rules.init_df(force=True)
        if db_examples:
            self.db_examples.init_df(force=True)
        if db_sites:
            self.db_sites.init_df(force=True)
        if _db_unknown:
            self.db_unknown.init_df(force=True)
        return True

    def rephrase_query(self, query) -> str:
        """
        takes query and returning it professional, and translate it to english if needed.
        it will slow down the system a little bit
        :param query: str
        :return: str
        """
        agent_message = self.agents_manager[GLOBALS.summarization_agent:query]
        return str(agent_message.agent_message)

    ######################################

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

    def set_rules(self, rule_types: list[dict] | None = None, _eval=False) -> bool:
        """
        Loads and embeds new rules into the rule database and agent.
        Args:
            rule_types (list[dict] | None): A list of dictionaries representing new rules.
        Returns:
            bool: True if the rules were successfully added and saved.
            :param _eval: if loading for evaluation.
        """

        # get all the fields and the queries to embed
        if _eval:
            rules_fields, chunks_to_embed = self.add_new_types.load(rule_types=rule_types,
                                                                    folder=GLOBALS.evaluation_rules_folder_path)
        else:
            rules_fields, chunks_to_embed = self.add_new_types.load(rule_types=rule_types)

        # agent embed and add everything to the agent data
        rules_names = [rule['rule_name'] for rule in rules_fields]
        rules_names, rules_embeddings = self.agents_manager[GLOBALS.classifier_agent].get_sampleS_embeddings(
            rules_names,
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
        rule_name, rule_embeddings = self.agents_manager[GLOBALS.classifier_agent].get_sample_embeddings(
            rule_fields["rule_name"], words_to_embed)
        rule_fields["embeddings"] = rule_embeddings

        # add to the db for future loading
        try:
            index = self.db_rules.df['rule_name'].tolist().index(rule_name)
            self.db_rules.df.loc[index] = rule_fields
        except ValueError:  # not exist yet in the data
            self.db_rules.df.loc[len(self.db_rules.df)] = rule_fields
        self.db_rules.save_db()
        return True

    def remove_rule(self, rule_name: str):
        """
        Removes a rule from the rule database and the agent based on its name.

        Args:
            rule_name (str): The name of the rule to remove.

        Returns:
            bool: True if the rule was successfully removed, False if the rule wasn't found.
        """
        if rule_name not in self.db_rules.df['rule_name'].values:
            return False

        # Remove the rule
        self.db_rules.df = self.db_rules.df[self.db_rules.df['rule_name'] != rule_name]

        # Reset the index after removal
        self.db_rules.df = self.db_rules.df.reset_index(drop=True)
        return True

    ######################################
    def set_sites(self, sites: list[dict]):
        """
        REMOVE PREVIOUS SITES!
        Loads and embeds new rules into the rule database and agent.
        Args:
            sites (list[dict]: A list of dictionaries representing new sites.
        Returns:
            bool: True if the sites were successfully added and saved.
        expected = {'site': 'ashdod', 'site_id': __site_id__}
        """

        # agent embed and add everything to the agent data
        sites_names = [site_dict["site"] for site_dict in sites]
        sites_names, sites_embeddings = self.agents_manager[GLOBALS.classifier_agent].get_sampleS_embeddings(
            sites_names, sites_names)
        for i in range(len(sites)):
            sites[i]["embeddings"] = sites_embeddings[i]

        # add to the db for future loading
        self.db_sites.df = pd.DataFrame(sites)
        self.db_sites.save_db()
        return True

    def add_site(self, site: dict):
        # expected = {'site': 'ashdod', 'site_id': __site_id__}
        rule_name, rule_embeddings = self.agents_manager[GLOBALS.classifier_agent].get_sample_embeddings(site["site"], site["site"])
        site["embeddings"] = rule_embeddings

        # add to the db for future loading
        try:
            index = self.db_sites.df['site'].tolist().index(rule_name)
            self.db_sites.df.loc[index] = site
        except ValueError:  # not exist yet in the data
            self.db_sites.df.loc[len(self.db_sites.df)] = site
        self.db_sites.save_db()
        return True

    def remove_site(self, site_name: str):
        if site_name not in self.db_sites.df['site'].values:
            return False

        # Remove the rule
        self.db_sites.df = self.db_sites.df[self.db_sites.df['site'] != site_name]

        # Reset the index after removal
        self.db_sites.df = self.db_sites.df.reset_index(drop=True)
        return True

    def get_existing_sites(self) -> list:
        return [{'site': site, 'site_id': site_id} for site, site_id in
                zip(self.db_sites.df['site'].tolist(), self.db_sites.df['site_id'].tolist())]
