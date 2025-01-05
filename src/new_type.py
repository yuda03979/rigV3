import json
import os
from .globals import GLOBALS


class AddNewType:
    """
    Prepares all the data for a single new rule type (without embedding).
    """

    folder = GLOBALS.rules_folder_path

    def add(self, rule_type: dict | str):
        """
        Adds a new rule type by preparing schema, description, and default values.

        :param rule_type: A dictionary representing the rule type or a string (file name) pointing to a JSON file.
        :return: A tuple containing:
                 - A dictionary with rule metadata and default rule instance.
                 - A string of words to embed for processing.
        :raises FileNotFoundError: If the provided file path does not exist.
        :raises json.JSONDecodeError: If the JSON file cannot be parsed.
        """

        if isinstance(rule_type, str):
            file_path = os.path.join(self.folder, rule_type)
            if file_path.endswith(".json"):
                with open(file_path, 'r') as file:
                    rule_type_json = file.read()
                rule_type: dict = json.loads(rule_type_json)

        rule_name = rule_type['name'].lower()
        schema = self.create_schema(rule_type)
        description = self.create_description(rule_type)
        default_values = self.create_default_values(rule_type)
        default_rule_instance = self.create_default_rule_instance(rule_type, default_values)

        words_to_embed = f"rule type name: {rule_name}\nschema: {schema}"

        return {
            'rule_name': rule_name,
            'schema': schema,
            'description': description,
            'default_values': default_values,
            'default_rule_instance': default_rule_instance,
            'rule_type': rule_type,
        }, words_to_embed

    def create_schema(self, rule_type: dict) -> dict:
        """
        Generates the schema for the rule type.

        :param rule_type: A dictionary representing the rule type with parameters.
        :return: A dictionary containing the schema with parameter names and their data types.
        """
        schema = {}
        for param in rule_type["parameters"]:
            schema[param["name"]] = str(param["type"])
        schema['ruleInstanceName'] = "string"
        schema['severity'] = "int"
        return schema

    def create_description(self, rule_type: dict) -> dict:
        """
        Generates descriptions for each rule parameter.

        :param rule_type: A dictionary representing the rule type with parameters and descriptions.
        :return: A dictionary containing descriptions for each parameter, global description, and event details.
        """
        description = {}
        for param in rule_type["parameters"]:
            description[param["name"] + "_description"] = str(param["description"])
        description['ruleInstanceName_description'] = "About what the message is and its relation to the database."
        description['severity_description'] = "Level of importance, criticality, or risk."
        description["event details"] = {}
        description["global description"] = str(rule_type["description"])
        description["object name"] = rule_type["eventDetails"][0]["objectName"]
        return description

    def create_default_values(self, rule_type: dict) -> dict:
        """
        Generates default values for each rule parameter.

        :param rule_type: A dictionary representing the rule type.
        :return: A dictionary with default values for each parameter.
        """
        default_values = {}
        for param in rule_type["parameters"]:
            default_values[param["name"]] = str(param["defaultValue"])
        return default_values

    def create_default_rule_instance(self, rule_type: dict, default_values: dict) -> dict:
        """
        Creates a default instance of the rule type.

        :param rule_type: A dictionary representing the rule type.
        :param default_values: A dictionary containing default values for rule parameters.
        :return: A dictionary representing the default rule instance with standard fields populated.
        """
        rule_instance = {
            "_id": "00000000-0000-0000-0000-000000000000",
            "description": "string",
            "isActive": True,
            "lastUpdateTime": "00/00/0000 00:00:00",
            "params": default_values,
            "ruleInstanceName": '',
            "severity": '',
            "ruleType": rule_type['logicType'],
            "ruleOwner": "",
            "ruleTypeId": rule_type["_id"],
            "eventDetails": rule_type["eventDetails"],
            "additionalInformation": rule_type['additionalInformation'],
            "presetId": "00000000-0000-0000-0000-000000000000"
        }
        return rule_instance


class AddNewTypes(AddNewType):
    """
    Prepares a batch of rule types for storage in the database (without embedding).
    """

    folder = GLOBALS.rules_folder_path

    def load(self, rule_types: list[dict] | None = None):
        """
        Loads rule types from the folder or accepts a list of rule types as input.

        :param rule_types: A list of rule type dictionaries, or None to load from the folder.
        :return: A tuple containing:
                 - A list of dictionaries representing prepared rule types.
                 - A list of strings to embed for each rule type.
        :raises FileNotFoundError: If the rule files are not found in the specified folder.
        :raises json.JSONDecodeError: If a rule file cannot be parsed.
        """
        # Create rule_types list if not provided
        if rule_types is None:
            rule_types = []
            for file_name in os.listdir(self.folder):
                file_path = os.path.join(self.folder, file_name)
                if file_path.endswith(".json"):
                    with open(file_path, 'r') as file:
                        rule_type_json = file.read()
                    rule_type: dict = json.loads(rule_type_json)
                    rule_types.append(rule_type)

        # For each rule, prepare all necessary fields except embeddings
        rules_fields = []
        chunks_to_embed = []  # list of what we will search similarity
        for rule in rule_types:
            fields = self.add(rule)
            rules_fields.append(fields[0])
            chunks_to_embed.append(fields[1])

        return rules_fields, chunks_to_embed
