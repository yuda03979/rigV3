from pydantic import BaseModel

from src.rig import Rig
from fastapi import FastAPI
from dotenv import find_dotenv, load_dotenv
import nest_asyncio

load_dotenv(find_dotenv())

app = FastAPI()
rig = Rig()
nest_asyncio.apply()

class Data(BaseModel):
    query: str

@app.post("/get_rule_instance")
def get_rule_instance(free_text) -> dict:
    return rig.get_rule_instance(free_text)


@app.post("/tweak_parameters")
def tweak_parameters(**kwargs) -> bool:
    kwargs = {k: v for k, v in kwargs.items()}
    return rig.tweak_parameters(**kwargs)


@app.post("/feedback")
def feedback(rig_response: dict, good: bool) -> bool:
    return rig.feedback(rig_response, good)


@app.post("/evaluate")
def evaluate(
        start_point: int = 0,
        end_point: int | None = 2,  # -1 - all the data
        jump: int | None = 1,
        sleep_time_each_10_iter: int = 30,
        batch_size: int = 250,
        set_eval_rules=True  # deleting existing rules!!! and loading the directory
) -> dict:
    return rig.evaluate(
        start_point=int(start_point),
        end_point=end_point,  # None - all the data
        jump=jump,
        sleep_time_each_10_iter=int(sleep_time_each_10_iter),
        batch_size=int(batch_size),
        set_eval_rules=set_eval_rules
    )


@app.get("/metadata")
def metadata() -> dict:
    return rig.metadata()


@app.post("/rephrase_query")
def rephrase_query(query: Data) -> dict:
    return dict(response=rig.rephrase_query(query.query))


@app.post("/restart")
def restart(db_rules=False, db_examples=False, db_sites=False, _db_unknown=False) -> bool:
    return rig.restart(db_rules=db_rules, db_examples=db_examples, db_sites=db_sites, _db_unknown=_db_unknown)


#############################


@app.get("/get_rules_names")
def get_rules_names() -> list[str]:
    return rig.get_rules_names()


@app.get("/get_rule_details")
def get_rule_details(rule_name: str) -> dict:
    return rig.get_rule_details(rule_name)


@app.post("/set_rules")
def set_rules() -> bool:
    return rig.set_rules()


@app.post("/add_rule")
def add_rule(json_file_name) -> bool:
    return rig.add_rule(json_file_name)

@app.post("/remove_rule")
def remove_rule(rule_name: str) -> bool:
    """
    API endpoint to remove a rule by its name.
    """
    result = rig.remove_rule(rule_name)
    return result


@app.post("/set_sites")
def set_sites(sites: list) -> bool:
    """
    API endpoint to add multiple sites.
    """
    return rig.set_sites(sites)


@app.post("/add_site")
def add_site(site: dict) -> bool:
    """
    API endpoint to add a single site.
    """
    return rig.add_site(site)


@app.post("/remove_site")
def remove_site(site_name: str) -> bool:
    """
    API endpoint to remove a site by its name.
    """
    result = rig.remove_site(site_name)
    return result


@app.get("/get_existing_sites")
def get_existing_sites() -> list:
    """
    API endpoint to get a list of existing sites.
    """
    return rig.get_existing_sites()
