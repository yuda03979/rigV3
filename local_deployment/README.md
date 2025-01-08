# getting started

### this is NOT for the docker - see README.Docker

----------------------
# Installation

ollama:
- for linux:
```angular2html
curl -fsSL https://ollama.com/install.sh | sh
```
- for other os check ollama website.
-------------
-------------
lfs:
```angular2html
git lfs install
```
- or:
```angular2html
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
```

## - downlads (if using internet):
gemma:
```
curl -L -O https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q8_0.gguf
```
rag:
```
https://huggingface.co/yixuan-chia/snowflake-arctic-embed-m-long-GGUF/resolve/main/snowflake-arctic-embed-m-long-F16.gguf?download=true
```
validation:
```
https://huggingface.co/tiiuae/Falcon3-3B-Instruct-GGUF/resolve/main/Falcon3-3B-Instruct-q4_k_m.gguf?download=true
```

and place them in the rig_modelfile directory (or change the path inside the modelfile. for docker its better they're together. and you can delete the file after - just keep the directory) 

but you also can download the models from the drive (there is also the modelfiles)
```angular2html
https://drive.google.com/drive/folders/1Jm97UnsVPvk_QpjnZi7ItNHHuqXsPhGq
```

if you using windows and NOT docker change the path inside the modelfile in the first line (after FROM)

------

## for docker check the README.docker

------------

## how to use:
on the terminal run
```angular2html
ollama serve
```

- now run in the load_gguf_modelfile the 2th cell.
- notice i assume you already set the models_directory in ollama.

-----------
# how to use
```python
from src.rig import Rig
rig = Rig()
```

```
rig.add_rule_types_from_folder()
```
```python
# get list of existing rule types:
rule_names = rig.get_rule_types_names()  # return list
```

```
rule_name = rule_names[0]  # Get the first rule name from the list
rule_details = rig.get_rule_details(rule_name)
```


### add rules
```
new_rules: list[dict] = [
    rule_1, rule_2
]

rig.set_rules(rule_types=new_rules)  # true or false
```

```
new_rule = {}
rig.add_rule(rule_type=new_rule)
```

## get rule instance from free text


```python
free_text = "Alright, let's dive in. We're looking at 'Exploitation Scenario 789'. The crux of the matter is, there's this individual, going by the ID 'XYZ789', who's been involved in an exploitation failure. The level of seriousness? I'd estimate about three. The breach? Fairness. Not good, not good at all. When did we spot this? Well, the detection time isn't clear. And the context? Personal. Yes, it's a pretty serious situation"
response = rig.get_rule_instance(free_text) # return dictionary
```


```python
response.keys()
```

dict_keys(['rule_instance', 'error', 'error_message', 'free_text', 'type_name', 'rag_score', 'model_response', 'schema', 'time', 'inference_time'])




```python
response["rule_instance"] # the package response
```


output:

    {'_id': '00000000-0000-0000-0000-000000000000',
     'description': 'string',
     'isActive': True,
     'lastUpdateTime': '00/00/0000 00:00:00',
     'params': {'individualID': 'XYZ789',
      'failureType': 'Fairness',
      'detectionTime': 'null',
      'ethicalViolation': 'null',
      'context': 'Personal'},
     'ruleInstanceName': 'Exploitation Scenario 789',
     'severity': 3,
     'ruleType': 'structured',
     'ruleOwner': '',
     'ruleTypeId': '7a2f6c94-2b4f-4d9d-8a77-d11f7c7cc8fc',
     'eventDetails': [{'objectName': 'Moral',
       'objectDescription': None,
       'timeWindowInMilliseconds': 0,
       'useLatest': False}],
     'additionalInformation': {},
     'presetId': '00000000-0000-0000-0000-000000000000'}


```python
rig.tweak_parameters(classification_threshold=0.05) 
```

```python
# giving us feedback on the response. it will help us to improve the project. it stores in .logs file, without internet connection.
rig.feedback(rig_response=response, good=False)
```

## run evaluation:
```
rig.evaluate(
    start_point=0,
    end_point=2,  # -1 or None - all the data
    jump=1,  # how many it skip and not evaluate
    sleep_time_each_10_iter=5,
    batch_size=250
)
```
