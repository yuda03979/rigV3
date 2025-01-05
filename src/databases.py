import ast
import json
from typing import Any
from .globals import GLOBALS
import pandas as pd


class DbBase:
    db_path = "remember to init that"
    df = "remember to init that"
    columns: list = ["remember to init that"]

    def init_df(self) -> None:
        self.validate_db_path(self.db_path)
        try:
            self.df = pd.read_csv(self.db_path)
            for col in self.columns:
                self.df[col] = self.df[col].apply(json.loads)
        except:
            self.df = pd.DataFrame(columns=self.columns)
            self.save_db()

    def validate_db_path(self, db_path) -> None:
        if not db_path.endswith('.csv'):
            message = f"the db_path should end with '.csv'! your path: {db_path}"
            raise ValueError(message)

    def save_db(self) -> None:
        for col in self.columns:
            self.df[col] = self.df[col].apply(json.dumps)
        self.df.to_csv(self.db_path, index=False)
        self.df = pd.read_csv(self.db_path)
        for col in self.columns:
            self.df[col] = self.df[col].apply(json.loads)


class DbRules(DbBase):
    db_path = GLOBALS.db_rules_path
    columns = ["rule_name", "schema", "description", "default_values", "default_rule_instance", "rule_type", "embeddings"]

    def __init__(self):
        self.init_df()


class DbExamples(DbBase):
    db_path = GLOBALS.db_examples_path
    columns = ["id", "free_text", "rule_name", "schema", "description", "rule_instance", "embeddings"]
    # in this point id and free text are the same
    def __init__(self):
        self.init_df()

    def get_data(self, example):
        return (
            self.df[self.df["id"] == example, "free_text"],
            self.df[self.df["id"] == example, "schema"],
            self.df[self.df["id"] == example, "rule_instance"],
        )
