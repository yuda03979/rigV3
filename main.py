from src.rule_instance_generator import Rig
from fastapi import FastAPI
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

app = FastAPI()
rig = Rig()


@app.post("/get_rule_instance")
def get_rule_instance(free_text) -> dict:
    return rig.get_rule_instance(free_text)


@app.get("/get_rule_types_names")
def get_rule_types_names() -> list:
    return rig.get_rule_types_names()


@app.get("/get_rule_type_details")
def get_rule_type_details(rule_name: str) -> dict:
    return rig.get_rule_type_details(rule_name)


@app.post("/set_rule_types")
def set_rule_types() -> None:
    return rig.set_rule_types()


@app.post("/add_rule_type")
def add_rule_type(json_file_name):
    return rig.add_rule_type(json_file_name)


@app.post("/tweak_parameters")
def tweak_parameters(
        rag_threshold=os.getenv("RAG_THRESHOLD")
) -> None:
    return rig.tweak_parameters(
        rag_threshold=float(rag_threshold)
    )


@app.post("/feedback")
def feedback(rig_response: dict, good: bool) -> dict:
    return rig.feedback(rig_response, good)


@app.post("/evaluate")
def evaluate(
        start_point=0,
        end_point=2,  # -1 - all the data
        sleep_time_each_10_iter=30,
        batch_size=250
) -> dict:
    print("evaluate")
    if end_point is not None:
        end_point = int(end_point)
    return rig.evaluate(
        start_point=int(start_point),
        end_point=end_point,  # None - all the data
        sleep_time_each_10_iter=int(sleep_time_each_10_iter),
        batch_size=int(batch_size)
    )