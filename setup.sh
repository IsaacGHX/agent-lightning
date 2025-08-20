#!/bin/bash

# Ensure script exits on error
set -e

# Switch to project root directory
cd /home/ubuntu/jianwen-us-midwest-1/panlu/ipf-projects/agent-lightning
source .venv/bin/activate

# Install UV (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "UV not installed, installing..."
    pip install uv
fi

# Create and activate UV virtual environment
if [ ! -d ".venv" ]
then
    echo "Creating UV virtual environment..."
    uv venv
fi

 echo "Activating virtual environment..."
source .venv/bin/activate

# Install project dependencies (development mode)
echo "Installing project dependencies (development mode)..."
uv pip install -e .[dev]

# Install additional dependency packages

echo "Installing AutoGen..."
uv pip install "autogen-agentchat" "autogen-ext[openai]"

echo "Installing LiteLLM..."
uv pip install "litellm[proxy]"

echo "Installing MCP..."
uv pip install mcp

echo "Installing OpenAI Agents..."
uv pip install openai-agents

echo "Installing LangChain related packages..."
uv pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters

echo "Installing SQL related dependencies..."
uv pip install sqlparse nltk

# Restart Ray service
echo "Restarting Ray service..."
bash scripts/restart_ray.sh

# Run calculator agent
echo "Running calculator agent..."
python examples/calc_x/calc_agent.py &
AGENT_PID=$!

echo "Waiting for agent to start..."
sleep 5