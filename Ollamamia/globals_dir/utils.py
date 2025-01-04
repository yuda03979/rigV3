from datetime import datetime
from typing import Optional, Union, Literal, get_origin, get_args, Sequence, Any
from collections.abc import Sequence as SequenceABC
from pydantic import BaseModel, Field
import uuid
import re
import ast
import json


def custom_is_instance(value, field_type):
    """Custom isinstance check for parameterized generics."""
    # Handle None for Optional types
    if value is None:
        if field_type is type(None):
            return True
        origin = get_origin(field_type)
        args = get_args(field_type)
        return origin is Union and type(None) in args

    origin = get_origin(field_type)
    args = get_args(field_type)

    # Simple types (no origin)
    if origin is None:
        if isinstance(field_type, type):
            return isinstance(value, field_type)
        return False

    # Handle Union types (including Optional)
    elif origin is Union:
        return any(
            isinstance(value, arg) if isinstance(arg, type) and not get_origin(arg)
            else custom_is_instance(value, arg)
            for arg in args
        )

    # Handle Sequence types
    elif origin is Sequence or (isinstance(origin, type) and issubclass(origin, SequenceABC)):
        if not isinstance(value, (list, tuple)):
            return False
        if not args:
            return True
        element_type = args[0]
        return all(
            isinstance(v, element_type) if isinstance(element_type, type) and not get_origin(element_type)
            else custom_is_instance(v, element_type)
            for v in value
        )

    # Handle list type
    elif origin is list:
        if not isinstance(value, list):
            return False
        if not args:
            return True
        element_type = args[0]
        return all(
            isinstance(v, element_type) if isinstance(element_type, type) and not get_origin(element_type)
            else custom_is_instance(v, element_type)
            for v in value
        )

    # Handle tuple type
    elif origin is tuple:
        if not isinstance(value, tuple):
            return False
        if not args:
            return True
        element_type = args[0]
        return all(
            isinstance(v, element_type) if isinstance(element_type, type) and not get_origin(element_type)
            else custom_is_instance(v, element_type)
            for v in value
        )

    # Handle dict type
    elif origin is dict:
        if not isinstance(value, dict):
            return False
        if not args:
            return True
        key_type, value_type = args
        return (
                all(
                    isinstance(k, key_type) if isinstance(key_type, type) and not get_origin(key_type)
                    else custom_is_instance(k, key_type)
                    for k in value.keys()
                ) and
                all(
                    isinstance(v, value_type) if isinstance(value_type, type) and not get_origin(value_type)
                    else custom_is_instance(v, value_type)
                    for v in value.values()
                )
        )

    # Handle Literal type
    elif origin is Literal:
        return value in args

    return False


class CustomBasePydantic(BaseModel):
    def __setattr__(self, name, value):
        # Perform validation on field assignment
        field_type = self.__annotations__.get(name)
        if field_type and not custom_is_instance(value, field_type) and value is not None:
            raise ValueError(f"Field '{name}' must be of type {field_type}, got {type(value)}.")
        super().__setattr__(name, value)


class AgentMessage(CustomBasePydantic):
    """
    All the agents must return only this object.
    """
    agent_name: str
    agent_description: str
    succeed: bool
    agent_input: str | dict | list | tuple
    agent_message: str | dict | list | tuple
    message_model: str | list
    dateTtime: datetime = Field(default_factory=datetime.now, alias="dateTtime")
    infer_time: float | None = None


class AgentsFlow(CustomBasePydantic):
    """
    In this object, we save all the data across the pipeline.
    """
    query: Any
    message: dict = {}
    is_error: bool = False  # change to success
    agents_massages: Optional[list[AgentMessage]] = None
    total_infer_time: float = -float("inf")
    Uuid: uuid.UUID = Field(default_factory=uuid.uuid4)
    dateTtime: datetime = Field(default_factory=datetime.now, alias="dateTtime")

    def append(self, agent_message: AgentMessage):
        if not self.agents_massages:
            self.agents_massages = [agent_message]
        else:
            self.agents_massages.append(agent_message)


def handle_errors(e: str):
    print(e)
    raise


def get_dict(input_string):
    # Use regex to find content between { and } that looks like a valid JSON
    input_string = fix_unbalanced_braces(input_string)
    input_string = re.sub(r"[\t\n]", "", input_string)
    match = re.search(r'\{[^}]*\}', input_string)

    if not match:
        return input_string, False

    json_str = match.group(0)

    try:
        # First, try standard JSON parsing
        parsed_dict = json.loads(json_str)
        return parsed_dict, True
    except json.JSONDecodeError:
        # If standard parsing fails, try some custom parsing
        try:
            json_str = re.sub(r"(None|null|'None'|\"None\"|'null'|\"null\")", '"null"', json_str)
            # Use ast for more flexible parsing
            parsed_dict = ast.literal_eval(json_str)
            return parsed_dict, True
        except (SyntaxError, ValueError):
            return input_string, False


def fix_unbalanced_braces(response):
    """
    Fix unbalanced braces in a model response by ensuring correct matching of { and }.
    """
    response = re.sub(r"[\t\n]", "", response)  # Remove tabs and newlines

    open_count = response.count('{')
    close_count = response.count('}')

    if close_count > open_count:
        excess_close = close_count - open_count
        response = response.replace('}', '', excess_close)

    elif open_count > close_count:
        missing_close = open_count - close_count
        response += '}' * missing_close  # Add missing closing braces

    return response
