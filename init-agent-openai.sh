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
docker compose up -d

# Parse command-line arguments
USER=""
MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            USER="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if both user and mode arguments are provided
if [[ -z "$USER" ]] || [[ -z "$MODE" ]]; then
    echo "Error: Both --user and --mode arguments are required."
    exit 1
fi

# Step 2: Run the Python script with user and mode arguments
echo "All checks passed. Running Python script with user=$USER and mode=$MODE..."
poetry run python3 main.py --user "$USER" --mode "$MODE" &  # Start Python script in the background
PYTHON_PID=$!  # Store the Python script's PID

# Keep the script running until the user presses Ctrl-C (SIGINT)
while true; do
    # Wait for the Python script to finish (checking its exit code)
    wait "$PYTHON_PID"
    PYTHON_EXIT_CODE=$?

    # If the Python process exits with an error, show the message and clean up
    if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
        echo "Python script exited with an error (exit code $PYTHON_EXIT_CODE)."
        cleanup
    fi
done
