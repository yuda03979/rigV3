
# Project Setup and Deployment Guide

## important!!!
- it will run slow on windows and mac, since docker is on vm
- linux its ok and fast.

### Installation Requirements
- lfs (for large file handling)
- Docker
- Docker Compose


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
wget https://huggingface.co/yixuan-chia/snowflake-arctic-embed-m-long-GGUF/resolve/main/snowflake-arctic-embed-m-long-F16.gguf?download=true -O snowflake-arctic-embed-m-long-F16.gguf
```

validation:
```
wget https://huggingface.co/tiiuae/Falcon3-3B-Instruct-GGUF/resolve/main/Falcon3-3B-Instruct-q4_k_m.gguf?download=true -O Falcon3-3B-Instruct-q4_k_m.gguf
```

smallm2
```
wget https://huggingface.co/lmstudio-community/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf?download=true -O SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```
remember to rename this model (it has dot in the middle...)

and place them in the rig_modelfile directory (or change the path inside the modelfile. for docker its better they're together. and you can delete the file after - just keep the directory) 


###### place this on the .env as GGUF_AND_MODELFILE_LOCATION

----------------
### 2. Configuration

#### Environment Configuration
1. edit the `.env` file
2. Update paths and settings as required
3. Ensure all necessary environment variables are set correctly

## Running the Application

install docker
```
sudo snap install docker  
```

install docker compose
```
sudo apt-get install docker-compose-plugin
```
or:
```angular2html
sudo curl -L "https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```
output: ```Docker Compose version v2.29.1```
### Local Development
Navigate to the project directory and run:

```bash
docker build -t rig .
```
```bash
docker compose up
```

The application will be accessible visually at: http://localhost:8000/docs


## the models:
after the project is up on docker, 
make the ollama models. 
in the load_gguf_modelfile.ipynb run the first cell.
- notice that you only need to do it once, and you also can delete the directory after (just keep the directory empty) 

# how to use:
it uses fastAPI.
(you also have how_to_docker.ipynb for python functions.)
- functions:
1. get_rule_instance
```
curl -X 'POST' \
  'http://0.0.0.0:80/get_rule_instance?free_text=We%20need%20to%20create%20an%20instance%20of%20Network%20Security%20Breach.%20The%20issue%20pertains%20to%20a%20network%20security%20breach.%20The%20coding%20efficiency%20is%20merely%20average%20and%20the%20problem-solving%20skills%20are%20good.%20Code%20readability,%20unfortunately,%20is%20poor.%20However,%20the%20team%20collaboration%20skills%20are%20excellent.%20The%20debugging%20expertise%20can%20be%20rated%20as%20high.%20The%20severity%20of%20this%20event?%20Well,%20it%27s%20a%20seven.%20As%20for%20the%20framework%20knowledge,%20it%27s...%20well,%20let%27s%20leave%20it%20blank%20for%20now.' \
  -H 'accept: application/json' \
  -d ''
```
another example
```
curl -X 'POST'   'http://0.0.0.0:80/get_rule_instance?free_text=A%20situation%20with%20the%20ruleInstanceName%20of%20rodent%20examination%20delta%20needs%20to%20be%20investigated.%20We%27re%20dealing%20with%20a%20severity%20level%20of%20one.%20The%20tail%20bushiness%20is%20bristly,%20and%20um,%20the%20hunting%20stealth%20is%20low.%20However,%20there%27s%20no%20information%20on%20den%20construction,%20let%27s%20just%20say%20it%27s%20null.%20The%20howling%20pitch%20is%20two,%20and%20the%20nocturnal%20habits%20are%20very%20active.%20The%20prey%20diversity%20is%20four,%20and%20the%20speed%20bursts%20are%20something%20like%20fifteen.'   -H 'accept: application/json'   -d ''
```
output:
dict with a lot of data...

```
and the same all the functions. see how_to_docker.ipynb
```



- **Documentation** (Swagger UI):
  Open a browser and navigate to:
  ```
  http://127.0.0.1:80/docs
  ```

- **Alternative Documentation** (ReDoc):
  ```
  http://127.0.0.1:80/redoc
  ```

5. **Use cURL or HTTP Clients to Test the Endpoints**
Here are examples of how to interact with each endpoint:

a. **POST `/get_rule_instance`**
```bash
curl -X POST "http://127.0.0.1:80/get_rule_instance" -H "Content-Type: application/json" -d '{"free_text": "example free text"}'
```

b. **GET `/get_rules_names`**
```bash
curl -X GET "http://127.0.0.1:80/get_rules_names"
```

c. **GET `/get_rule_details`**
```bash
curl -X GET "http://127.0.0.1:80/get_rule_details?rule_name=example_rule_name"
```

d. **POST `/set_rules`**
```bash
curl -X POST "http://127.0.0.1:80/set_rules"
```

e. **POST `/add_rule`**
```bash
curl -X POST "http://127.0.0.1:80/add_rule" -H "Content-Type: application/json" -d '{"json_file_name": "rule.json"}'
```

f. **POST `/tweak_parameters`**
```bash
curl -X POST "http://127.0.0.1:80/tweak_parameters" -H "Content-Type: application/json" -d '{"classification_threshold": 0.9, "classification_temperature": 0.7}'
```

g. **POST `/feedback`**
```bash
curl -X POST "http://127.0.0.1:80/feedback" -H "Content-Type: application/json" -d '{"rig_response": {"query": "example query"}, "good": true}'
```

h. **POST `/evaluate`**
```bash
curl -X 'POST'   'http://0.0.0.0/evaluate?start_point=0&end_point=3&jump=1&sleep_time_each_10_iter=30&batch_size=250&set_eval_rules=true'   -H 'accept: application/json'   -d ''
```

i. **GET `/metadata`**
```bash
curl -X GET "http://127.0.0.1:80/metadata"
```

j. **POST `/restart`**
```bash
curl -X POST "http://127.0.0.1:80/restart" -H "Content-Type: application/json" -d '{"db_rules": true, "db_examples": true}'
```

k. **POST `/rephrase_query`**
```bash
curl -X POST "http://127.0.0.1:80/rephrase_query" -H "Content-Type: application/json" -d '{"query": "example query"}'
```

6. **Stop the Server**
To stop the running server, press `Ctrl+C` in the terminal.
