import json
import os
from .globals import GLOBALS


class AddNewType:
    """
    prepare all the data for single new rule type (no embedding)
    """
    folder = GLOBALS.rules_folder_path

    def add(self, rule_type: dict | str):
        """

        :param rule_type: the file_name or the dict
        :return: the data for using the rule
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

    def create_schema(self, rule_type) -> dict:
        schema = {}
        for param in rule_type["parameters"]:
            schema[param["name"]] = str(param["type"])
        schema['ruleInstanceName'] = "string"
        schema['severity'] = "int"
        return schema

    def create_description(self, rule_type) -> dict:
        description = {}
        for param in rule_type["parameters"]:
            description[param["name"] + "_description"] = str(param["description"])
        description['ruleInstanceName_description'] = "about what the message and to what it related in the db."
        description['severity_description'] = "level of importance, criticality, or risk."
        description["event details"] = {}
        description["global description"] = str(rule_type["description"])
        description["object name"] = rule_type["eventDetails"][0]["objectName"]
        return description

    def create_default_values(self, rule_type) -> dict:
        """ assuming severity and ruleInstanceName default values"""
        default_values = {}
        for param in rule_type["parameters"]:
            default_values[param["name"]] = str(param["defaultValue"])
        return default_values

    def create_default_rule_instance(self, rule_type, default_values) -> dict:
        rule_instance = {
            "_id": "00000000-0000-0000-0000-000000000000",  # Sample ID
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
    prepare batch of rule types for storing in db. (no embedding)
    """

    folder = GLOBALS.rules_folder_path

    def load(self, rule_types: list[dict] | None):

        #  create rule_types: list[dict] of rules.
        if rule_types is None:
            rule_types = []
            for file_name in os.listdir(self.folder):
                file_path = os.path.join(self.folder, file_name)
                if file_path.endswith(".json"):
                    with open(file_path, 'r') as file:
                        rule_type_json = file.read()
                    rule_type: dict = json.loads(rule_type_json)
                    rule_types.append(rule_type)

        #  for each rule create everything its need except the embeddings
        rules_fields = []
        chunks_to_embed = []
        for rule in rule_types:
            fields = self.add(rule)
            rules_fields.append(fields[0])
            chunks_to_embed.append(fields[1])

        return rules_fields, chunks_to_embed

