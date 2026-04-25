# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands
- **Install Dependencies**: `uv sync` (or `pip install .` for core dependencies)
- **Build Docker Image**: `docker build -t my_env-env:latest -f server/Dockerfile .`
- **Run Server Locally**: `uvicorn server.app:app --reload` or `uv run --project . server`
- **Test Environment Logic**: `python3 server/cloud_arena_environment.py` (Note: Ensure you use the correct environment file, `cloud_arena_environment.py` is the main logic)
- **Deploy to Hugging Face**: `openenv push`

## Architecture Overview
The project is a multi-agent "Cloud Resource Negotiation Arena" built on the OpenEnv framework.

### Core Components
- **Server (`server/`)**: 
  - `cloud_arena_environment.py`: The core environment logic implementing the negotiation arena.
  - `app.py`: FastAPI application providing HTTP and WebSocket endpoints for the environment.
  - `gradio_dashboard.py`: Web UI for interacting with the arena and visualizing state.
  - `negotiation_engine.py`, `market_dynamics.py`, `trust_manager.py`: Specialized logic for the arena's mechanics.
- **Agents (`agents/`)**: 
  - Specialized agent types (e.g., `devops_agent.py`, `ml_agent.py`, `rogue_agent.py`) inheriting from `base_agent.py`.
- **Learning & Metrics (`learning/`, `metrics/`)**: 
  - Tools for cost estimation, trust graphs, value modeling, and performance visualization.
- **Training (`training/`)**: 
  - Infrastructure for training agents using TRL and Transformers.

### Data Flow
1. **Client** sends an `ArenaAction` to the server via WebSocket or HTTP.
2. **Server** (`CloudArenaEnvironment`) processes the action through the `negotiation_engine` and `market_dynamics`.
3. **Server** returns an `ArenaObservation` and a reward.
4. **Dashboard** provides a real-time view of the cluster state and negotiation progress.
