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
    def __init__(self):
        self.run_async_models = True if os.getenv("RUN_ASYNC_MODELS").lower() == "true" else False
        self.max_examples = validate_numeric("MAX_EXAMPLES", value_type=int)

        self.summarization_agent = "summarization"
        self.classifier_agent = "rule_classifier"
        self.rule_instance_generator_agent = "rule_instance_generator"

        self.generation_model_name = os.getenv("GENERATION_MODEL_NAME")
        self.validation_model_name = os.getenv("VALIDATION_MODEL_NAME")
        self.rag_model_name = os.getenv("RAG_MODEL_NAME")

        self.project_dir = validate_path("PROJECT_DIR")
        self.eval_dir = validate_path("EVAL_DIR")
        self.classification_threshold = validate_numeric("CLASSIFICATION_THRESHOLD", value_type=float)
        self.site_rag_threshold = validate_numeric("SITE_RAG_THRESHOLD", value_type=float)
        self.add_example_rag_threshold = validate_numeric("ADD_EXAMPLE_RAG_THRESHOLD", value_type=float)
        self.rag_temperature = validate_numeric("RAG_TEMPERATURE", value_type=float)

        # things that should be in project_dir
        self.db_rules_path = os.path.join(self.project_dir, "db_rules.csv")
        self.db_examples_path = os.path.join(self.project_dir, "db_examples.csv")
        self.db_unknowns_path = os.path.join(self.project_dir, "db_unknowns.csv")
        self.db_site_path = os.path.join(self.project_dir, "db_sites.csv")

        self.rules_folder_path = os.path.join(self.project_dir, "rule_types")

        # evaluation data
        self.evaluation_rules_folder_path = os.path.join(self.eval_dir, "rule_types")
        self.evaluation_data_path = os.path.join(self.eval_dir, "evaluation_data.csv")
        self.evaluation_output_dir = os.path.join(self.eval_dir, "output")


GLOBALS = Globals()
