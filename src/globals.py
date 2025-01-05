import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def validate_path(var_name):
    value = os.getenv(var_name)
    if not value or not os.path.exists(value):
        print(f"{var_name} is not set or the path does not exist: {value}")
    return value

def validate_numeric(var_name, value_type):
    value = os.getenv(var_name)
    if value is None:
        print(f"{var_name} is not set")
    try:
        return value_type(value)
    except ValueError:
        raise ValueError(f"{var_name} must be a valid {value_type.__name__}")


class Globals:

    rule_classifier_agent = "rule_classifier"
    examples_finder_agent = "examples_finder"
    rule_instance_generator_agent = "rule_instance_generator"

    project_dir = validate_path("PROJECT_DIR")

    db_rules_path = os.path.join(project_dir, "db_rules.csv")
    db_examples_path = os.path.join(project_dir, "db_examples.csv")

    rules_folder_path = os.path.join(project_dir, "eval", "rule_types")
    evaluation_data_path = os.path.join(project_dir, "eval", "evaluation_data.csv")


GLOBALS = Globals()
