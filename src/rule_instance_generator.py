import pydantic
from .globals import GLOBALS
from .post_processing import post_processing


class RuleInstanceGenerator:
    """
    A class that generates rule instances based on free text input using a combination of
    classification, generation, and post-processing steps.
    """

    def predict(self, agents_manager, db_rules, db_examples, free_text):
        """
        Main entry point for rule instance prediction.

        Args:
            agents_manager: Manager object handling agent interactions
            db_rules: Database containing rule definitions and schemas
            db_examples: Database containing example rule instances
            free_text: String input to generate rules from

        Returns:
            dict: Response containing the generated rule instance or error information
        """
        response = self.generate(agents_manager, db_rules, db_examples, free_text)
        return response

    def generate(self, agents_manager, db_rules, db_examples, free_text):
        """
        Generates a rule instance through classification and generation steps.

        Args:
            agents_manager: Manager object handling agent interactions
            db_rules: Database containing rule definitions and schemas
            db_examples: Database containing example rule instances
            free_text: String input to generate rules from

        Returns:
            dict: Response containing the generated rule instance or error information
                 Contains keys: is_error, error_message, confidence, rule_instance
        """
        rule_names_list = self.__classify_rule(agents_manager, db_rules, free_text)
        if rule_names_list is None:
            return dict(is_error=True, error_message="didn't find rule name")

        rule_name = rule_names_list[0][0]
        response, mismatch_rule_name = self.__generate(agents_manager, db_rules, db_examples, rule_name, free_text)

        if mismatch_rule_name and len(rule_names_list) >= 2:
            # print(rule_names_list[:2])
            rule_name = rule_names_list[1][0]
            response2, mismatch_rule_name = self.__generate(agents_manager, db_rules, db_examples, rule_name, free_text)

            if not mismatch_rule_name:
                print("solved")
                response = response2
            else:
                print("unsolved 2th attempt")
                response["confidence"] = -1
                response["is_error"] = True
                response["error_message"] = ("since more then 50% of the fields are None/'null', we assume its "
                                             "classification mismatch, and we can't solve it")
        elif mismatch_rule_name and len(rule_names_list) < 2:
            print("unsolved 1th attempt")
            response["confidence"] = -2
            response["is_error"] = True
            response["error_message"] = ("since more then 50% of the fields are None/'null', we assume its "
                                         "classification mismatch, and we can't solve it")
        return response

    def __generate(self, agents_manager, db_rules, db_examples, rule_name, free_text):
        """
        Internal method to generate a rule instance for a specific rule name.

        Args:
            agents_manager: Manager object handling agent interactions
            db_rules: Database containing rule definitions and schemas
            db_examples: Database containing example rule instances
            rule_name: Name of the rule to generate an instance for
            free_text: String input to generate rules from

        Returns:
            tuple: (response_dict, mismatch_rule_name_bool)
                  response_dict contains the generated rule instance or error information
                  mismatch_rule_name_bool indicates if the rule name was mismatched
        """
        schema, description, default_rule_instance = self.__get_params(db_rules, rule_name)
        example1, example2 = self.__get_examples(agents_manager, db_examples, free_text)
        response, success = self.__generate_with_schema(agents_manager, free_text, schema, rule_name, example1,
                                                        example2, description)

        if success:
            response, mismatch_rule_name = self.__post_processing(response, rule_name, schema, default_rule_instance)
        else:
            response, mismatch_rule_name = None, False

        return response, mismatch_rule_name

    def __classify_rule(self, agents_manager, db_rules, free_text):
        """
        Classifies the input text to determine appropriate rule names.

        Args:
            agents_manager: Manager object handling agent interactions
            free_text: String input to classify

        Returns:
            list: List of tuples containing (rule_name, score) pairs, sorted by score
        """
        agent_message: pydantic.BaseModel = agents_manager[GLOBALS.rule_classifier_agent].predict(
            query=free_text,
            samples_id=db_rules.df["rule_name"].tolist(),
            samples_embeddins=db_rules.df["embeddings"].tolist()
        )
        return agent_message.agent_message

    def __get_params(self, db_rules, rule_name):
        """
        Retrieves rule parameters from the rules database.

        Args:
            db_rules: Database containing rule definitions
            rule_name: Name of the rule to get parameters for

        Returns:
            tuple: (schema, description, default_rule_instance)
        """
        schema = db_rules.df.loc[db_rules.df["rule_name"] == rule_name, "schema"].iloc[0]
        description = db_rules.df.loc[db_rules.df["rule_name"] == rule_name, "description"].iloc[0]
        default_rule_instance = db_rules.df.loc[db_rules.df["rule_name"] == rule_name, "default_rule_instance"].iloc[0]
        return schema, description, default_rule_instance

    def __get_examples(self, agents_manager, db_examples, free_text):
        """
        Retrieves relevant examples for rule generation.

        Args:
            agents_manager: Manager object handling agent interactions
            db_examples: Database containing example rule instances
            free_text: String input to find examples for

        Returns:
            tuple: (example1, example2) where each example is a formatted string containing
                  free_text, schema, and output information
        """
        agent_message: pydantic.BaseModel = agents_manager[GLOBALS.examples_finder_agent:free_text]

        example1 = agent_message.agent_message[0]
        example2 = agent_message.agent_message[1]

        if example1 is not None and example2 is not None:
            example1_free_text, example1_schema, example1_output = db_examples.get_example(example1)
            example2_free_text, example2_schema, example2_output = db_examples.get_example(example2)

            example1 = f"Free_text:\n{example1_free_text},\nSchema:\n{example1_schema},\nOutput:\n{example1_output}"
            example2 = f"Free_text:\n{example2_free_text},\nschema:\n{example2_schema},\nOutput:\n{example2_output}"

        return example1, example2

    def __generate_with_schema(self, agents_manager, free_text, schema, rule_name, example1, example2, description):
        """
        Generates a rule instance using the provided schema and examples.

        Args:
            agents_manager: Manager object handling agent interactions
            free_text: String input to generate from
            schema: Schema definition for the rule
            rule_name: Name of the rule being generated
            example1: First example for reference
            example2: Second example for reference
            description: Description of the rule

        Returns:
            tuple: (response_dict, success_bool)
                  response_dict contains the generated rule instance and metadata
                  success_bool indicates if generation was successful
        """
        agent_message: pydantic.BaseModel = agents_manager[GLOBALS.rule_instance_generator_agent:dict(
            query=free_text,
            schema=str(schema),
            rule_name=rule_name,
            example1=example1,
            example2=example2,
            description=str(description)
        )]

        agents_flow: pydantic.BaseModel = agents_manager.get_agents_flow()

        response = agents_flow.model_dump()
        response["rule_name"] = rule_name
        response["rule_instance_params"] = agent_message.agent_message[0]
        response["confidence"] = agent_message.agent_message[1]
        response["error_message"] = ""

        success = not agents_flow.is_error
        return response, success

    def __post_processing(self, response, rule_name, schema, default_rule_instance):
        """
        Performs post-processing on the generated rule instance.

        Args:
            response: Dictionary containing the generated rule instance and metadata
            rule_name: Name of the rule being processed
            schema: Schema definition for the rule
            default_rule_instance: Default instance to fall back on

        Returns:
            tuple: (response_dict, mismatch_rule_name_bool)
                  response_dict contains the processed rule instance
                  mismatch_rule_name_bool indicates if rule name was mismatched
        """
        response["rule_instance"], mismatch_rule_name = post_processing(
            rule_name,
            response["rule_instance_params"],
            schema=schema,
            default_rule_instance=default_rule_instance
        )
        return response, mismatch_rule_name