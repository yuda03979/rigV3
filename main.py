from pydantic import BaseModel, Field
from fastapi import FastAPI, Body
from typing import Optional, List, Dict, Any, Union
from dotenv import find_dotenv, load_dotenv
import nest_asyncio
from src.rig import Rig

load_dotenv(find_dotenv())
app = FastAPI()
rig = Rig()
nest_asyncio.apply()


# Request/Response Models
class QueryData(BaseModel):
    query: str = Field(description="Query string to process")

class SetRulesRequest(BaseModel):
    rules: Optional[List[Dict[str, Any]]] = None


class ParametersTweakModel(BaseModel):
    run_async_models: bool = Field(
        description="Whether to run models asynchronously"
    )
    classification_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Threshold for classification confidence"
    )
    site_rag_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Threshold for site RAG similarity"
    )
    rag_temperature: float = Field(
        ge=0.0, le=2.0,
        description="Temperature parameter for RAG generation"
    )
    add_example_rag_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Threshold for adding examples in RAG"
    )
    max_examples: int = Field(
        ge=0,
        description="Maximum number of examples to include"
    )


class FeedbackModel(BaseModel):
    rig_response: Dict[str, Any] = Field(..., description="The response from the RIG system")
    good: bool = Field(..., description="Whether the feedback is positive")


class EvaluationParams(BaseModel):
    start_point: int = Field(default=0, ge=0, description="Starting point for evaluation")
    end_point: Optional[int] = Field(default=2, description="Ending point for evaluation")
    jump: Optional[int] = Field(default=1, gt=0, description="Step size for evaluation")
    sleep_time_each_10_iter: int = Field(default=30, ge=0, description="Sleep time after every 10 iterations")
    batch_size: int = Field(default=250, gt=0, description="Batch size for evaluation")
    set_eval_rules: bool = Field(default=True, description="Whether to set evaluation rules")


class Site(BaseModel):
    site: str = Field(..., description="Site name, e.g., 'ashdod'")
    site_id: Any = Field(..., description="Unique identifier for the site, e.g., '1234'")


class RestartParams(BaseModel):
    db_rules: bool = Field(default=False, description="Whether to restart rules database")
    db_examples: bool = Field(default=False, description="Whether to restart examples database")
    db_sites: bool = Field(default=False, description="Whether to restart sites database")
    db_unknown: bool = Field(default=False, description="Whether to restart unknown database")


# Endpoints
@app.post("/get_rule_instance")
def get_rule_instance(free_text) -> dict:
    result = rig.get_rule_instance(free_text)
    return result


@app.post("/tweak_parameters")
def tweak_parameters(params: ParametersTweakModel) -> bool:
    return rig.tweak_parameters(**params.model_dump())


@app.post("/feedback")
def feedback(feedback_data: FeedbackModel) -> bool:
    return rig.feedback(feedback_data.rig_response, feedback_data.good)


@app.post("/evaluate")
def evaluate(params: EvaluationParams) -> dict:
    return rig.evaluate(
        start_point=params.start_point,
        end_point=params.end_point,
        jump=params.jump,
        sleep_time_each_10_iter=params.sleep_time_each_10_iter,
        batch_size=params.batch_size,
        set_eval_rules=params.set_eval_rules
    )


@app.get("/metadata")
def metadata() -> dict:
    return rig.metadata()


@app.post("/rephrase_query")
def rephrase_query(query: QueryData) -> dict:
    return dict(response=rig.rephrase_query(query.query))


@app.post("/restart")
def restart(params: RestartParams) -> bool:
    return rig.restart(**params.model_dump())


@app.get("/get_rules_names", response_model=List[str])
def get_rules_names() -> List[str]:
    return rig.get_rules_names()


@app.get("/get_rule_details")
def get_rule_details(rule_name: str) -> dict:
    result = rig.get_rule_details(rule_name)
    return result


@app.post("/set_rules")
def set_rules(request: SetRulesRequest = Body(default=None)) -> bool:
    return rig.set_rules(request.rules if request else None)


@app.post("/add_rule")
def add_rule(rule: str | dict) -> bool:
    if isinstance(rule, dict):
        if rule.get('name') is None:
            rule = rule['rule']
    result = rig.add_rule(rule)
    return result


@app.post("/remove_rule")
def remove_rule(rule_name: str) -> bool:
    result = rig.remove_rule(rule_name)
    return result


@app.post("/set_sites")
def set_sites(sites: List[Dict[str, Any]] = Body(...)) -> bool:
    print(sites)
    return rig.set_sites(sites)


@app.post("/add_site")
def add_site(site: Site) -> bool:
    return rig.add_site(site.model_dump())


@app.post("/remove_site")
def remove_site(site_name) -> bool:
    result = rig.remove_site(site_name)
    return result


@app.get("/get_existing_sites", response_model=List[Site])
def get_existing_sites() -> List[Site]:
    return rig.get_existing_sites()
