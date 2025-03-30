## 1. How to run

You should have an OpenAI api key which you should put in your `.env` file under the variable `OPENAI_API_KEY`.

Make sure you have installed `ffmpeg`, if not follow step 2 before proceeding. Also make sure you have installed [Docker](https://docs.docker.com/engine/install/).

You can start the agent by running the init script:

```zsh
./init-agent-openai.sh --user <userID> --mode <memory-mode>
```

The script should (mostly) be fool-proof and it won't allow you to run stuff until everything is in place.
The arguments should initiate a correct session for a particular user. This will also be the userID stored in the DB.

Memory mode can be either `semantic` or `episodic`.

The script should start a flask server with a front-end that you can access by navigating to [http://localhost:6555/](http://localhost:6555/)

You should see all the logs in `app.log` file created at the root of the project.

The script should also take care of shutting down the docker containers so you don't have to worry about it later.

## 2. Install ffmpeg for perception module

1. Make sure to install ffmpeg on your laptop for perception module to work.

   on Ubuntu or Debian

   sudo apt update && sudo apt install ffmpeg

   on Arch Linux

   sudo pacman -S ffmpeg

   on MacOS using Homebrew (https://brew.sh/)

   brew install ffmpeg

   on Windows using Chocolatey (https://chocolatey.org/)

   choco install ffmpeg

   on Windows using Scoop (https://scoop.sh/)

   scoop install ffmpeg

## 3. Setting Up MongoDB for Development

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your own user-name and password, rest should be kept the same.
3. Run `docker compose up -d` to set up the container and the database.

## TODO: Add more details on how to work with poetry and run modules

Example command to run the semantic workflow:

```bash
poetry run python3 -m binge_buddy.memory_workflow.semantic_workflow
```

Poetry will run the script inside the virtual environment.

You can also activate the virtualenv created by poetry yourself and then run

```bash
python3 -m binge_buddy.memory_workflow.semantic_workflow
```

These essentially do the same thing.

## 4. Steps needed to run deepseek locally

1. Install Opalla
2. Pull and run whatever model you want to run. Use the ollama pull command to download the DeepSeek model you want to use (e.g. `ollama pull deepseek-r1:8b`) and then run the model with (e.g. `ollama run deepseek-r1:8b`)
3. Start Opalla server with `ollama serve` to expose the model as an API
