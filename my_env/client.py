# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Client

"""
EnvClient for the Cloud Resource Negotiation Arena.
Connects to the environment server via WebSocket.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    ArenaObservation, ArenaAction, ActionType, ResourceProfile, AgentStatus, MarketState,
    Task, TaskCompletion, TaskProgress, CoalitionProposal,
)


class CloudArenaClient(EnvClient[ArenaAction, ArenaObservation, State]):
    """
    Client for the Cloud Resource Negotiation Arena.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> client = CloudArenaClient(base_url="http://localhost:8000").sync()
        >>> with client:
        ...     result = client.reset()
        ...     action = PassAction(agent_id="frontend")
        ...     result = client.step(action)
        ...     print(result.observation.current_round)
    """

    def _step_payload(self, action: ArenaAction) -> Dict[str, Any]:
        """Serialize action to JSON for the server."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ArenaObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})

        # Parse nested models carefully
        observation = ArenaObservation(
            current_step=obs_data.get("current_step", 0),
            current_round=obs_data.get("current_round", 0),
            phase=obs_data.get("phase", 1),
            observing_agent=obs_data.get("observing_agent", ""),
            unassigned_tasks=[Task(**t) for t in obs_data.get("unassigned_tasks", [])],
            cluster_utilization=ResourceProfile(**obs_data.get("cluster_utilization", {})),
            utilization_forecast=[
                ResourceProfile(**f) for f in obs_data.get("utilization_forecast", [])
            ],
            agent_roster=[AgentStatus(**a) for a in obs_data.get("agent_roster", [])],
            recent_completions=[
                TaskCompletion(**c) for c in obs_data.get("recent_completions", [])
            ],
            pending_proposals=[
                CoalitionProposal(**p) for p in obs_data.get("pending_proposals", [])
            ],
            market_conditions=MarketState(**obs_data.get("market_conditions", {})),
            my_tasks_in_flight=[
                TaskProgress(**t) for t in obs_data.get("my_tasks_in_flight", [])
            ],
            my_resource_budget=ResourceProfile(**obs_data.get("my_resource_budget", {})),
            trust_scores=obs_data.get("trust_scores", {}),
            active_alerts=obs_data.get("active_alerts", []),
            oversight_report=obs_data.get("oversight_report"),
            negotiation_log=obs_data.get("negotiation_log", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server state response."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )