#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo -e "\nShutting down..."
    
    # Kill the Python process if it's running
    if [[ -n "$PYTHON_PID" ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        kill "$PYTHON_PID"
        wait "$PYTHON_PID" 2>/dev/null
    fi

    # Shut down Docker Compose
    docker compose down

    # Shut down Ollama server
    pkill -f "ollama serve"

    exit 0
}

trap cleanup INT

# Step 1: Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

if ! (docker info &> /dev/null); then
    echo "Docker daemon is not running. Please start Docker and try again."
    exit 1
fi

# Run Docker Compose
echo "Starting Docker Compose..."
docker compose up

# Step 2: Check if Ollama is installed and running
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install Ollama and try again."
    docker compose down
    exit 1
fi

if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Ollama server is not running. Attempting to start..."
    ollama serve &> ollama.log &  # Run Ollama in the background and log output
    OLLAMA_PID=$!  # Store Ollama PID

    sleep 2  # Give some time for the server to start
    if ! pgrep -f "ollama serve" > /dev/null; then
        echo "Ollama server failed to start. Exiting."
        docker compose down
        exit 1
    fi
fi

# Step 3: Run the Python script and keep it running
echo "All checks passed. Running Python script..."
poetry run python3 main.py &  # Start Python script in the background
PYTHON_PID=$!  # Store the Python script's PID

wait "$PYTHON_PID"
