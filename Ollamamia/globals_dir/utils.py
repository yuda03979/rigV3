from datetime import datetime
from typing import Optional, Union, Literal, get_origin, get_args, Sequence, Any
from collections.abc import Sequence as SequenceABC
from pydantic import BaseModel, Field
import uuid
import re
import ast
import json

"""
This module implements a type-safe message passing system for agent-based workflows using Pydantic models
and custom type validation. It provides classes and utilities for handling agent messages, managing workflow
state, and performing type checking with support for complex Python type hints.
"""


def custom_is_instance(value, field_type):
    """
    Performs runtime type checking for complex Python type hints including generics and special types.

    Args:
        value: Any value to check the type of
        field_type: A type hint to check against (can be simple types, Union, Optional,
                   Sequence, List, Tuple, Dict, or Literal)

    Returns:
        bool: True if the value matches the type hint, False otherwise

    Examples:
        custom_is_instance(None, Optional[str])  # Returns True
        custom_is_instance([1, 2, 3], Sequence[int])  # Returns True
        custom_is_instance("invalid", Literal["valid", "also_valid"])  # Returns False
    """
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
    """
    A custom Pydantic base model that implements runtime type checking for field assignments.

    This class extends Pydantic's BaseModel to add runtime type validation when fields are
    set after instance creation. It uses custom_is_instance() for type checking to support
    complex Python type hints.

    Raises:
        ValueError: If a field is assigned a value of incorrect type
    """

    def __setattr__(self, name, value):
        # Perform validation on field assignment
        field_type = self.__annotations__.get(name)
        if field_type and not custom_is_instance(value, field_type) and value is not None:
            raise ValueError(f"Field '{name}' must be of type {field_type}, got {type(value)}.")
        super().__setattr__(name, value)


class AgentMessage(CustomBasePydantic):
    """
    A standardized message format that all agents must use for communication.

    This class represents the structure of messages that agents can exchange in the system.
    It includes metadata about the agent, success status, and the actual message content.

    Attributes:
        agent_name (str): Name identifier for the agent
        agent_description (str): Description of the agent's role or purpose
        succeed (bool): Whether the agent's operation was successful
        agent_input (Any): The input received by the agent
        agent_message (Any): The message/output produced by the agent
        message_model (Any): The model used to generate the message
        dateTtime (datetime): Timestamp of message creation (auto-generated)
        infer_time (float | None): Time taken for inference, if applicable
        additional_data (Any): Optional additional data the agent wants to include
    """
    agent_name: str
    agent_description: str
    succeed: bool
    agent_input: Any
    agent_message: Any
    message_model: Any
    dateTtime: datetime = Field(default_factory=datetime.now, alias="dateTtime")
    infer_time: float | None = None
    additional_data: Any = None


class AgentsFlow(CustomBasePydantic):
    """
    Manages the state and message flow of an agent-based pipeline.

    This class tracks the complete state of a workflow including all messages passed
    between agents, total processing time, and error states.

    Attributes:
        query (Any): The initial query or input to the pipeline
        message (dict): Additional message data, defaults to empty dict
        is_error (bool): Error state flag, defaults to False
        agents_massages (Optional[list[AgentMessage]]): List of all agent messages in sequence
        total_infer_time (float): Total inference time across all agents
        Uuid (uuid.UUID): Unique identifier for the flow (auto-generated)
        dateTtime (datetime): Timestamp of flow creation (auto-generated)
    """
    query: Any
    message: dict = {}
    is_error: bool = False  # change to success
    agents_massages: Optional[list[AgentMessage]] = None
    total_infer_time: float = -float("inf")
    Uuid: uuid.UUID = Field(default_factory=uuid.uuid4)
    dateTtime: datetime = Field(default_factory=datetime.now, alias="dateTtime")

    def append(self, agent_message: AgentMessage):
        """
        Adds a new agent message to the flow's history.

        Args:
            agent_message (AgentMessage): The message to append to the flow
        """
        if not self.agents_massages:
            self.agents_massages = [agent_message]
        else:
            self.agents_massages.append(agent_message)


def handle_errors(e: str):
    """
    Central error handling function for the agent system.

    Args:
        e (str): Error message or description

    Note:
        Currently only prints the error but could be extended to implement more
        sophisticated error handling strategies.
    """
    print(e)
    # raise


def get_dict(input_string: str) -> tuple[Union[dict, str], bool]:
    """
    Attempts to extract and parse a dictionary from a string containing JSON-like content.

    Args:
        input_string (str): String that may contain a dictionary-like structure

    Returns:
        tuple: (parsed_content, success_flag)
            - parsed_content: Either the parsed dictionary or the original string if parsing failed
            - success_flag: Boolean indicating whether parsing was successful

    Note:
        Handles various edge cases including unbalanced braces and alternative
        representations of null values.
    """
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


def fix_unbalanced_braces(response: str) -> str:
    """
    Fixes unbalanced braces in a string by adding or removing braces as needed.

    Args:
        response (str): String that may contain unbalanced braces

    Returns:
        str: String with balanced braces

    Note:
        Removes tabs and newlines before processing and ensures equal numbers
        of opening and closing braces.
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


def compare_dicts(dict1, dict2):
    NULL_VALUES = [
        None, '', ' ', "null", "None", "none", "empty", "undefined", "nil",
        "NaN", "nan", "n/a", "N/A", "na", "NA", "missing", "unknown", "void",
        "blank", ".", "..", "...", "?", "int", "Int", "String", "string"
    ]

    def normalize(value):
        if value in NULL_VALUES:
            return 'null'
        return str(value).strip().lower()

    # Create copies
    d1 = {k: normalize(v) for k, v in dict1.items()}
    d2 = {k: normalize(v) for k, v in dict2.items()}
    print('model: ', d1)
    print('model2: ', d2)
    return d1 == d2
