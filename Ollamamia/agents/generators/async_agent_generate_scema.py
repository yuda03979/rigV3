from Ollamamia.globals_dir.models_manager import ASYNC_MODEL_MANAGER
from Ollamamia.globals_dir.utils import get_dict, AgentMessage
import asyncio
import time


class AsyncAgentGenerateSchema:
    description = """given schema and free text, the agent job is to return the values from the free text according to the schema"""

    model_1th_nickname = str(ASYNC_MODEL_MANAGER.get_num_models())
    model_2th_nickname = f"2th_{str(ASYNC_MODEL_MANAGER.get_num_models())}"
    engine = "ollama"
    model_1th_name = "gemma-2-2b-it-Q8_0:rig"
    model_2th_name = "Falcon3-3B-Instruct-q4_k_m:rig"
    model_type = "gemma2"
    task = "generate"

    num_ctx = 2048
    stop = ["}"]
    temperature = 0.0
    top_p = 1.0

    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.model_1th_nickname = f"{agent_name}_{self.model_1th_nickname}"
        self.prompt_func = Prompts(self.engine, self.model_type).prompt_func

        # Initialize models
        self._init_models()

    def _init_models(self):
        for nickname, model_name in [(self.model_1th_nickname, self.model_1th_name),
                                     (self.model_2th_nickname, self.model_2th_name)]:
            ASYNC_MODEL_MANAGER[nickname] = [self.engine, model_name, self.task]
            model_config = ASYNC_MODEL_MANAGER[nickname].config.options
            model_config.num_ctx = self.num_ctx
            model_config.stop = self.stop
            model_config.temperature = self.temperature
            model_config.top_p = self.top_p

    async def predict_async(self, kwargs):
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

        # Run model inferences in parallel
        responses = await asyncio.gather(
            ASYNC_MODEL_MANAGER.infer_async(self.model_1th_nickname, prompt),
            ASYNC_MODEL_MANAGER.infer_async(self.model_2th_nickname, prompt)
        )

        response_model, response_model_2th = [r + "}" for r in responses]
        response, succeed = get_dict(response_model)
        response_2th, succeed_2th = get_dict(response_model_2th)


        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=[response, response_2th],
            message_model=response_model,
            infer_time=time.time() - start
        )
        return agent_message

    # Keep sync version for compatibility
    def predict(self, kwargs):
        return asyncio.run(self.predict_async(kwargs))


# Prompts class remains unchanged

class Prompts:

    def __init__(self, engine, model_1th_name):
        self.prompt_func = None
        if model_1th_name == "gemma2":
            self.prompt_func = self.prompt_gemma2

    @staticmethod
    def prompt_gemma2(free_text, rule_name, schema, description, example1=None, example2=None):
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
