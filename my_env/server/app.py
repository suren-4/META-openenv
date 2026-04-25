# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — FastAPI Application

"""
FastAPI application for the Cloud Resource Negotiation Arena.

Exposes the CloudArenaEnvironment over HTTP and WebSocket endpoints,
with an integrated Gradio dashboard at /web.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - /web: Gradio web interface with Playground + Arena Dashboard tabs

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server import create_web_interface_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from my_env.models import ArenaAction, ArenaObservation
    from my_env.server.cloud_arena_environment import CloudArenaEnvironment
    from my_env.server.gradio_dashboard import build_arena_dashboard
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import ArenaAction, ArenaObservation
        from .cloud_arena_environment import CloudArenaEnvironment
        from .gradio_dashboard import build_arena_dashboard
    except (ImportError, ModuleNotFoundError):
        from models import ArenaAction, ArenaObservation
        from server.cloud_arena_environment import CloudArenaEnvironment
        from server.gradio_dashboard import build_arena_dashboard


# Create the app with web interface
# Note: ArenaAction covers all action types (BID, PASS, etc.)
app = create_web_interface_app(
    CloudArenaEnvironment,
    ArenaAction,  # Primary action class for schema/UI
    ArenaObservation,
    env_name="cloud_resource_negotiation_arena",
    max_concurrent_envs=4,
    gradio_builder=build_arena_dashboard,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

    Usage:
        uv run --project . server
        python -m my_env.server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
