
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
and place them in the rig_modelfile directory (or change the path inside the modelfile. for docker its better they're together. and you can delete the file after - just keep the directory) 

but you also can download the models from the drive (there is also the modelfiles)
```angular2html
https://drive.google.com/drive/folders/1Jm97UnsVPvk_QpjnZi7ItNHHuqXsPhGq
```

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
  'http://0.0.0.0:8000/get_rule_instance?free_text=system%20failure%20severity%205' \
  -H 'accept: application/json' \
  -d ''
```
output:
{}

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
curl -X POST "http://127.0.0.1:80/evaluate" -H "Content-Type: application/json" -d '{"start_point": 0, "end_point": 2, "jump": 1, "batch_size": 250}'
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