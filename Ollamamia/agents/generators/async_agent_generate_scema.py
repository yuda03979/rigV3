from Ollamamia.globals_dir.models_manager import ASYNC_MODEL_MANAGER
from Ollamamia.globals_dir.utils import get_dict, AgentMessage, compare_dicts
import asyncio
import time
from src.globals import GLOBALS


class AsyncAgentGenerateSchema:
    """
    An asynchronous agent that extracts structured data from free text according to a given schema.

    This agent uses two language models in parallel for generation and validation, comparing their
    outputs to assess confidence. It supports both synchronous and asynchronous prediction methods.

    Attributes:
        description (str): Description of the agent's purpose
        model_1th_nickname (str): Identifier for the primary generation model
        model_2th_nickname (str): Identifier for the validation model
        engine (str): Engine used for models ("ollama")
        model_1th_name (str): Name of primary model from GLOBALS
        model_2th_name (str): Name of validation model from GLOBALS
        model_type (str): Type of model being used ("gemma2")
        task (str): Type of task ("generate")
        num_ctx (int): Context window size (2048)
        stop (list[str]): Stop tokens (["}"])
        temperature (float): Generation temperature (0.0)
        top_p (float): Top-p sampling parameter (1.0)
        model_name (list[str]): Names of both models
        model_nickname (list[str]): Nicknames of both models
    """

    description = """given schema and free text, the agent job is to return the values from the free text according to the schema"""

    model_1th_nickname = f"AsyncAgentGenerateSchema_{GLOBALS.generation_model_name}"
    model_2th_nickname = f"AsyncAgentGenerateSchema_{GLOBALS.validation_model_name}"
    engine = "ollama"
    model_1th_name = GLOBALS.generation_model_name  # "gemma-2-2b-it-Q8_0:rig"
    model_2th_name = GLOBALS.validation_model_name  # "Falcon3-3B-Instruct-q4_k_m:rig"
    model_type = "gemma2"
    task = "generate"

    num_ctx = 2048
    stop = ["}"]
    temperature = 0.0
    top_p = 1.0

    model_name = [GLOBALS.generation_model_name, GLOBALS.validation_model_name]
    model_nickname = [model_1th_nickname, model_2th_nickname]

    def __init__(self, agent_name: str):
        """
        Initialize the AsyncAgentGenerateSchema.

        Args:
            agent_name (str): Name identifier for the agent instance
        """
        self.agent_name = agent_name
        self.model_1th_nickname = f"{agent_name}_{self.model_1th_nickname}"
        self.prompt_func = Prompts(self.engine, self.model_type).prompt_func

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize both models with their configurations."""
        for nickname, model_name in [(self.model_1th_nickname, self.model_1th_name),
                                     (self.model_2th_nickname, self.model_2th_name)]:
            ASYNC_MODEL_MANAGER[nickname] = [self.engine, model_name, self.task]
            model_config = ASYNC_MODEL_MANAGER[nickname].config.options
            model_config.num_ctx = self.num_ctx
            model_config.stop = self.stop
            model_config.temperature = self.temperature
            model_config.top_p = self.top_p

    async def predict_async(self, kwargs: dict) -> AgentMessage:
        """
        Asynchronously process free text according to the provided schema.

        Args:
            kwargs (dict): Dictionary containing:
                - query (str): The free text to process
                - schema (dict): The target schema structure
                - rule_name (str, optional): Name of the rule being applied
                - example1 (dict[dict], optional): First example for prompt
                - example2 (dict[dict], optional): Second example for prompt
                - description (dict | str, optional): Additional context

        Returns:
            AgentMessage: Message object containing:
                - agent_name: Name of the agent
                - agent_description: Description of the agent
                - agent_input: Original query
                - succeed: Whether processing succeeded
                - agent_message: [processed_response, confidence]
                - message_model: [raw_model1_response, raw_model2_response]
                - infer_time: Processing time
                - additional_data: Details about both models' responses
        """
        query: str = kwargs["query"]
        schema: dict = kwargs["schema"]
        rule_name: str | None = kwargs.get("rule_name")
        example1: dict[dict] | None = kwargs.get("example1")
        example2: dict[dict] | None = kwargs.get("example2")
        description: dict | str | None = kwargs.get("description")

        start = time.time()

        prompt = self.prompt_func(
            free_text=query,
            rule_name=rule_name,
            schema=schema,
            description=description,
            example1=example1,
            example2=example2,
        )

        # prompt2 = (f"extract information as json according the schema. if field missing, then None."
        #            f"stick to the schema! no extra fields! just fill the schema"
        #            f": schema: {schema}, text: {query}: ```json")
        prompt2 = prompt + ("you should extract the information as json: ```json Here is the output for the given "
                            "schema + free text:\n")

        # Run model inferences in parallel
        responses = await asyncio.gather(
            ASYNC_MODEL_MANAGER.infer_async(self.model_1th_nickname, prompt),
            ASYNC_MODEL_MANAGER.infer_async(self.model_2th_nickname, prompt2)
        )

        response_model, response_model_2th = [r + "}" for r in responses]
        response, succeed = get_dict(response_model)
        response_2th, succeed_2th = get_dict(response_model_2th)



        additional_data = dict(
            model1=dict(dict_response=response, succeed=succeed, str_response=response_model).copy(),
            model2=dict(dict_response=response_2th, succeed=succeed_2th, str_response=response_model_2th),
            response="model1"
        )

        #######
        # confidence
        confidence = -1

        if succeed_2th and succeed:
            if compare_dicts(response, response_2th):
                confidence = 1
            else:
                confidence = 0

        # if the second model succeed and the first didn't we overwrite the first one:
        elif succeed_2th and not succeed:
            response = response_2th
            succeed = succeed_2th
            additional_data["response"] = "model2"

        ########

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=[response, confidence],
            message_model=[response_model, response_model_2th],
            infer_time=time.time() - start,
            additional_data=additional_data
        )
        return agent_message

    def predict(self, kwargs: dict) -> AgentMessage:
        """
        Synchronous wrapper for predict_async.

        Args:
            kwargs (dict): Same arguments as predict_async

        Returns:
            AgentMessage: Same return value as predict_async
        """
        return asyncio.run(self.predict_async(kwargs))


class Prompts:
    """
    Manages prompt generation for different model types.
    Currently supports the Gemma2 model type.
    """

    def __init__(self, engine: str, model_1th_name: str):
        """
        Initialize the Prompts manager.

        Args:
            engine (str): The engine being used
            model_1th_name (str): The model type for prompt selection
        """
        self.prompt_func = None
        if model_1th_name == "gemma2":
            self.prompt_func = self.prompt_gemma2

    @staticmethod
    def prompt_gemma2(free_text: str, rule_name: str, schema: dict, description: str | dict,
                      example1: dict | None = None, example2: dict | None = None) -> str:
        """
        Generate a prompt for the Gemma2 model.

        Args:
            free_text (str): The text to process
            rule_name (str): Name of the rule being applied
            schema (dict): The target schema structure
            description (str | dict): Additional context
            example1 (dict, optional): First example for the prompt
            example2 (dict, optional): Second example for the prompt

        Returns:
            str: The formatted prompt text
        """
        if not example1:
            example1 = """ 
            Free text:
            "Add a report for 'Shell Delay'. Equipment Malfunction case. Type: shell. Site is not important. Malfunction at level five, urgency four. Desc: Detonation delayed in poland Severity i think 3."
            Schema:
            {
                "type": "string",
                "site": "string",
                "malfunctionLevel": "int",
                "urgency": "int",
                "description": " string",
                "ruleInstanceName": "string",
                "severity": "int"
            }
            Output:
            {
                "type": "shell",
                "site": "empty",
                "malfunctionLevel": 5,
                "urgency": 4,
                "description": "Detonation delayed in poland",
                "ruleInstanceName": "Equipment Malfunction - Shell Delay",
                "severity": 3
            }"""
        if not example2:
            example2 = """"""

        ### Example 3 (Example for handling empty values):
        example3 = """
        Free text:
        "Please generate a Weather Alert - other for area code 2001. Alert type is Thunderstorm. Intensity: severe. Urgency: high. Ignore the forecast source for now. The duration of this case, I'd say, is around two. severity is empty or unclear"
        Schema:
        {
            "alertType": "string",
            "areaCode": "int",
            "intensity": "string",
            "urgency": "string",
            "forecastSource": "string",
            "duration": "int",
            "ruleInstanceName": "string",
            "severity": "int"
        }
        Output:
        {
            "alertType": "Thunderstorm",
            "areaCode": 2001,
            "intensity": "severe",
            "urgency": "high",
            "forecastSource": "null",
            "duration": 2,
            "ruleInstanceName": "Weather Alert - other",
            "severity": "null"
        }
        """
        examples = f"""
        ### Example 1 (Similar Style Example):
        {example1}

        ### Example 2 (Closest Task Match) :
        {example2}

        ### Example 3 (Example for handling empty values):
        """

        prompt = f"""
        ### Rules:
        1. Only use fields explicitly listed in the schema below.
        2. Do not add or infer fields. Fields missing in the schema or text should be set to "null".
        3. Carefully map field names from the schema to the text, even if the phrasing differs.
        4. Treat words like "without", "unknown" as "null", while "standard" "low" etc. will return the value.
        5. Output must match the schema exactly.

        ***Be sure to follow these rules***

        ### Examples:
        {examples}
        ---

        ### Context (do not use for filling fields, only as reference for the schema):
        {description}

        ---
        ### Task:
        - Schema: {schema}
        - Free text: {free_text}
        - Output:
        """
        # print("prompt = " + prompt)
        return prompt
