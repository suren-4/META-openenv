# GEMINI.md - Cloud Resource Negotiation Arena

## Project Overview
The **Cloud Resource Negotiation Arena** is a multi-agent environment built on the `openenv-core` framework. It simulates a Kubernetes cluster where four AI agents (Frontend, ML, Data, and DevOps) negotiate for shared resources (CPU, RAM, GPU, Bandwidth) to complete tasks. The project is specifically designed to facilitate research into multi-agent interactions and to train Large Language Models (LLMs) using Reinforcement Learning techniques like GRPO via the Hugging Face `trl` library.

### Core Architecture
- **Environment (`CloudArenaEnvironment`)**: A round-robin environment where each `step()` processes one agent's action. A "round" is resolved after all four agents have acted.
- **Server**: A FastAPI-based server that provides both a REST API and a WebSocket interface for low-latency agent interactions.
- **Dashboard**: An integrated Gradio UI (accessible via `/web`) that provides real-time visualization of task queues, cluster utilization, negotiation logs, and agent rewards.
- **Agents**: Supports both rule-based baselines (Greedy, Conservative, Coalition) and LLM-powered agents that interact via structured natural language prompts.
- **Training**: Includes a dedicated pipeline for TRL/GRPO training, allowing LLMs to improve their negotiation strategies through self-play.

## Key Technologies
- **Python 3.10+**
- **openenv-core**: Base environment framework.
- **FastAPI / Uvicorn**: Server-side runtime.
- **Gradio**: Interactive web dashboard.
- **Hugging Face TRL / Transformers**: LLM training and inference.
- **Plotly / Matplotlib**: Metrics and visualization.

## Building and Running

### Prerequisites
Ensure you have `uv` installed for dependency management, or use `pip`.
```bash
# Sync dependencies
uv sync
```

### Running the Server
```bash
# Start the environment server with auto-reload
uvicorn server.app:app --reload --port 8000
```
Access the dashboard at `http://localhost:8000/web`.

### Running Training
```bash
# Execute the TRL training script
python training/trl_training.py
```

### Building Docker Image
```bash
docker build -t cloud-arena:latest -f server/Dockerfile .
```

## Project Structure
- `agents/`: Implementation of various agent strategies (baselines + LLM).
- `server/`: Core environment logic including the cluster simulator, negotiation engine, and reward system.
- `models.py`: Pydantic models for `ArenaAction`, `ArenaObservation`, and cluster state.
- `learning/`: Components for trust graphs, cost estimation, and value modeling.
- `metrics/`: Tools for collecting and visualizing episode performance.
- `training/`: Scripts and notebooks for RL training.

## Development Conventions
- **Round-Robin Stepping**: Each `step()` corresponds to exactly one agent's turn. The environment tracks `current_agent_idx` and only resolves the round when all agents have submitted an action.
- **Structured Observations**: Observations include a `to_prompt()` method designed to convert complex state into natural language for LLM consumption.
- **Component Separation**: Logic is strictly modularized (e.g., `ClusterState` manages resources, `NegotiationEngine` handles bids, `RewardCalculator` handles scoring).
- **Testing**: Use `pytest` for unit testing components. Empirical validation via `training_runner.py` is recommended for behavioral changes.
