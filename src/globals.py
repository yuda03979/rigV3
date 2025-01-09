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
    # those 4 are just a nickname:
    summarization_agent = "summarization"
    rule_classifier_agent = "rule_classifier"
    examples_finder_agent = "examples_finder"
    rule_instance_generator_agent = "rule_instance_generator"

    generation_model_name = os.getenv("GENERATION_MODEL_NAME")
    validation_model_name = os.getenv("VALIDATION_MODEL_NAME")
    rag_model_name = os.getenv("RAG_MODEL_NAME")

    project_dir = validate_path("PROJECT_DIR")
    eval_dir = validate_path("EVAL_DIR")
    classification_threshold = validate_numeric("CLASSIFICATION_THRESHOLD", value_type=float)
    classification_temperature = validate_numeric("CLASSIFICATION_TEMPERATURE", value_type=float)
    examples_rag_threshold = validate_numeric("EXAMPLES_RAG_THRESHOLD", value_type=float)

    # things that should be in project_dir
    db_rules_path = os.path.join(project_dir, "db_rules.csv")
    db_examples_path = os.path.join(project_dir, "db_examples.csv")
    db_unknowns_path = os.path.join(project_dir, "db_unknowns.csv")

    rules_folder_path = os.path.join(project_dir, "rule_types")

    # evaluation data
    evaluation_rules_folder_path = os.path.join(eval_dir, "rule_types")
    evaluation_data_path = os.path.join(eval_dir, "evaluation_data.csv")
    evaluation_output_dir = os.path.join(eval_dir, "eval", "output")


GLOBALS = Globals()
