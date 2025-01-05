
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
https://huggingface.co/yixuan-chia/snowflake-arctic-embed-m-long-GGUF/resolve/main/snowflake-arctic-embed-m-long-F16.gguf?download=true
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
docker compose build
```
```angular2html
docker compose up
```

The application will be accessible visually at: http://localhost:8000/docs


## the models:
after the project is up on docker, 
make the ollama models. 
in the load_gguf_modelfile.ipynb run the first cell.
- notice that you only need to do it once, and you also can delete the directory after (just keep the directory empty) 

# how to use:
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
and the same all the functions. see how to docker.ipynb
```
