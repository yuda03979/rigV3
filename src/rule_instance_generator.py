import pydantic
from .globals import GLOBALS
from .post_processing import post_processing


class RuleInstanceGenerator:

    def __init__(self):
        pass

    def predict(self, agents_manager, db_rules, db_examples, free_text):
        response = self.generate(agents_manager, db_rules, db_examples, free_text)
        response["confidence"] = -2
        if response.get("confidence") == -2:  # we couldn't classify the rule
            summarize_query = agents_manager[GLOBALS.summarization_agent:free_text].agent_message
            response = self.generate(agents_manager, db_rules, db_examples, free_text, summarize_query)
        return response

    def generate(self, agents_manager, db_rules, db_examples, free_text, summarize_query=None):
        #######
        # classify the rule name. in the first time we don't summarize for saving time
        if not summarize_query:
            summarize_query = free_text
        rule_names_list = self.__classify_rule(agents_manager, summarize_query)
        if rule_names_list is None:
            return dict(is_error=True, error_message="didn't find rule name")
        rule_name = rule_names_list[0][0]

        ########
        # first try:
        response, mismatch_rule_name = self.__generate(agents_manager, db_rules, db_examples, rule_name, free_text)

        #######
        # in case of wrong classification:
        if mismatch_rule_name and len(rule_names_list) >= 2:
            print(rule_names_list[:2])
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
        #######
        # get params for the generation part
        schema, description, default_rule_instance = self.__get_params(db_rules, rule_name)

        #######
        # get examples
        example1, example2 = self.__get_examples(agents_manager, db_examples, free_text)

        #######
        # generate the rule instance
        response, success = self.__generate_with_schema(agents_manager, free_text, schema, rule_name, example1,
                                                        example2, description)

        ########
        # post processing
        if success:
            response, mismatch_rule_name = self.__post_processing(response, rule_name, schema, default_rule_instance)
        else:
            response, mismatch_rule_name = None, False

        return response, mismatch_rule_name

    def __classify_rule(self, agents_manager, free_text):
        """
        :param agents_manager:
        :param free_text:
        :return: list of all the rule_names with scores. [(rule_name, score), ...]
        """
        agent_message: pydantic.BaseModel = agents_manager[GLOBALS.rule_classifier_agent:free_text]

        rule_names_list = agent_message.agent_message
        return rule_names_list

    def __get_params(self, db_rules, rule_name):
        # with the rule_name get from the db_rules the schema and description

        schema = db_rules.df.loc[db_rules.df["rule_name"] == rule_name, "schema"].iloc[0]
        description = db_rules.df.loc[db_rules.df["rule_name"] == rule_name, "description"].iloc[0]
        default_rule_instance = db_rules.df.loc[db_rules.df["rule_name"] == rule_name, "default_rule_instance"].iloc[0]

        return schema, description, default_rule_instance

    def __get_examples(self, agents_manager, db_examples, free_text):
        agent_message: pydantic.BaseModel = agents_manager[GLOBALS.examples_finder_agent:free_text]

        example1 = agent_message.agent_message[0]
        example2 = agent_message.agent_message[1]

        if example1 is not None or example2 is not None:
            example1_free_text, example1_schema, example1_output = db_examples.get_example(example1)
            example2_free_text, example2_schema, example2_output = db_examples.get_example(example2)

            example1 = f"Free_text:\n{example1_free_text},\nSchema:\n{example1_schema},\nOutput:\n{example1_output}"
            example2 = f"Free_text:\n{example2_free_text},\nschema:\n{example2_schema},\nOutput:\n{example2_output}"

        return example1, example2

    def __generate_with_schema(self, agents_manager, free_text, schema, rule_name, example1, example2, description) -> \
            tuple[dict, bool]:

        agent_message: pydantic.BaseModel = agents_manager[GLOBALS.rule_instance_generator_agent:dict(
            query=free_text,
            schema=str(schema),
            rule_name=rule_name,
            example1=example1,
            example2=example2,
            description=str(description)
        )]

        agents_flow: pydantic.BaseModel = agents_manager.get_agents_flow()

        # print(agent_message)

        response = agents_flow.model_dump()
        response["rule_name"] = rule_name
        response["rule_instance_params"] = agent_message.agent_message[0]
        response["confidence"] = agent_message.agent_message[1]
        response["error_message"] = ""

        success = not agents_flow.is_error
        return response, success

    def __post_processing(self, response, rule_name, schema, default_rule_instance):
        response["rule_instance"], mismatch_rule_name = post_processing(
            rule_name,
            response["rule_instance_params"],
            schema=schema,
            default_rule_instance=default_rule_instance
        )
        return response, mismatch_rule_name
