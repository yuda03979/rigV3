from ...globals_dir.models_manager import MODELS_MANAGER
from ...globals_dir.utils import get_dict
import time
from ...globals_dir.utils import AgentMessage
from src.globals import GLOBALS


class AgentGenerateSchema:
    description = """given schema and free text, (and maybe some more parameters - depends on the prompt you choose
    the agent job is to return the values from the free text according to the schema"""

    model_nickname = str(MODELS_MANAGER.get_num_models())
    engine = "ollama"
    model_name = GLOBALS.generation_model_name  # "gemma-2-2b-it-Q8_0:rig"
    model_type = "gemma2"
    task = "generate"

    num_ctx = 2048
    stop = ["}"]
    temperature = 0.0
    top_p = 1.0

    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.model_nickname = f"{agent_name}_{self.model_nickname}"
        self.prompt_func = Prompts(self.engine, self.model_type).prompt_func
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.options.num_ctx = self.num_ctx
        MODELS_MANAGER[self.model_nickname].config.options.stop = self.stop
        MODELS_MANAGER[self.model_nickname].config.options.temperature = self.temperature
        MODELS_MANAGER[self.model_nickname].config.options.top_p = self.top_p

    def predict(
            self,
            kwargs  # need change. i did it because it get dict
    ):
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

        response_model = MODELS_MANAGER[self.model_nickname].infer(prompt) + "}"
        response, succeed = get_dict(response_model)

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=response,
            message_model=response_model,
            infer_time=time.time() - start
        )
        return agent_message


class Prompts:

    def __init__(self, engine, model_name):
        self.prompt_func = None
        if model_name == "gemma2":
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
