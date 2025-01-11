import ast
import json
from typing import Any
from .globals import GLOBALS
import pandas as pd


class DbBase:
    """
    Base class for database operations, providing common functionality for managing CSV-based databases.
    """
    db_path = "remember to init that"
    df = "remember to init that"
    columns: list = ["remember to init that"]

    def init_df(self, force: bool = False) -> None:
        """
        Initializes the DataFrame by reading the database file from the specified path.
        Ensures that the specified columns are parsed as JSON. If the database file does not exist,
        creates an empty DataFrame with the specified columns and saves it.
        """
        self.validate_db_path(self.db_path)
        try:
            self.df = pd.read_csv(self.db_path)
            for col in self.columns:
                self.df[col] = self.df[col].apply(json.loads)
        except:
            self.df = pd.DataFrame(columns=self.columns)
            self.save_db()

        if force:
            self.df = pd.DataFrame(columns=self.columns)
            self.save_db()

    def validate_db_path(self, db_path) -> None:
        """
        Validates the database path to ensure it ends with '.csv'.

        Args:
            db_path (str): The path to the database file.

        Raises:
            ValueError: If the database path does not end with '.csv'.
        """
        if not db_path.endswith('.csv'):
            message = f"the db_path should end with '.csv'! your path: {db_path}"
            raise ValueError(message)

    def save_db(self) -> None:
        """
        Saves the current DataFrame to the database file. Ensures that specified columns
        are serialized as JSON before saving. Reads the saved file back into the DataFrame
        and deserializes the JSON columns.
        """
        for col in self.columns:
            self.df[col] = self.df[col].apply(json.dumps)
        self.df.to_csv(self.db_path, index=False)
        self.df = pd.read_csv(self.db_path)
        for col in self.columns:
            self.df[col] = self.df[col].apply(json.loads)


class DbRules(DbBase):
    """
    Class for managing the rules database. Inherits from DbBase and specifies
    the database path and columns for rules.
    """
    db_path = GLOBALS.db_rules_path
    columns = ["rule_name", "schema", "description", "default_values", "default_rule_instance", "rule_type",
               "embeddings"]

    def __init__(self):
        """
        Initializes the rules database by loading the data from the specified database path.
        """
        self.init_df()


class DbExamples(DbBase):
    """
    Class for managing the examples database. Inherits from DbBase and specifies
    the database path and columns for examples.
    """
    db_path = GLOBALS.db_examples_path
    columns = ["id", "free_text", "rule_name", "schema", "description", "rule_instance_params", "embeddings", 'usage']

    def __init__(self):
        """
        Initializes the examples database by loading the data from the specified database path.
        """
        self.init_df()
        self.max_examples = GLOBALS.max_examples

    def get_example(self, example):
        """
        Retrieves data for a specific example ID from the database.

        Args:
            example: The ID of the example to retrieve.

        Returns:
            tuple: A tuple containing free_text, schema, and rule_instance for the specified example ID.
        """
        # Find the row with matching ID without using set_index
        matching_row = self.df[self.df['id'] == example]

        if len(matching_row) > 0:
            # Get the first matching row
            row = matching_row.iloc[0]
            return row['free_text'], row['schema'], row['rule_instance']
        return None, None, None

    def remove_unused(self):
        try:
            if self.max_examples < len(self.df):
                usage_numeric = pd.to_numeric(self.df['usage'])
                min_usage = usage_numeric.min()
                min_usage_records = self.df[usage_numeric == min_usage]
                oldest_record_index = min_usage_records.index[0]
                self.df = self.df.drop(oldest_record_index)
                print("One example deleted")
        except:
            return False
        return True



class DbUnknowns(DbBase):
    """
    Class for managing the examples database. Inherits from DbBase and specifies
    the database path and columns for examples.
    """
    db_path = GLOBALS.db_unknowns_path
    columns = ['query', 'message', 'is_error', 'agents_massages', 'total_infer_time', 'Uuid', 'dateTtime', 'rule_name',
               'rule_instance_params', 'confidence', 'error_message', 'rule_instance', 'good']

    def __init__(self):
        """
        Initializes the examples database by loading the data from the specified database path.
        """
        self.init_df()


class DbSites(DbBase):
    """
    Class for managing the examples database. Inherits from DbBase and specifies
    the database path and columns for examples.
    """
    db_path = GLOBALS.db_site_path
    columns = ["site", "site_id", "embeddings"]

    def __init__(self):
        """
        Initializes the examples database by loading the data from the specified database path.
        """
        self.init_df()
