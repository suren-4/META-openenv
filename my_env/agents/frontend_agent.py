# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Frontend Agent

"""
Opportunistic baseline agent.
Focuses on quick, high-value tasks (like API deployments and emergency patches).
"""

try:
    from my_env.models import ArenaObservation, ArenaAction, ActionType
except ImportError:
    try:
        from ..models import ArenaObservation, ArenaAction, ActionType
    except (ImportError, ValueError):
        from models import ArenaObservation, ArenaAction, ActionType
from .base_agent import BaseAgent


class FrontendAgent(BaseAgent):
    """Opportunistic agent strategy."""

    def act(self, observation: ArenaObservation) -> ArenaAction:
        # Simple strategy: Find a task matching my team affinity and bid on it
        my_tasks = [t for t in observation.unassigned_tasks if t.primary_team == self.team]
        
        if my_tasks:
            # Sort by value (descending)
            my_tasks.sort(key=lambda t: t.base_value, reverse=True)
            target = my_tasks[0]
            
            # Simple bid: ask for the base resources, offer 100% of price, ETA = deadline
            return ArenaAction(
                action_type=ActionType.BID,
                agent_id=self.agent_id,
                task_id=target.id,
                resource_request=target.resources_required,
                price_offered=target.base_value,
                eta_minutes=target.deadline_minutes,
                confidence=0.9
            )
            
        return ArenaAction(action_type=ActionType.PASS, agent_id=self.agent_id)
