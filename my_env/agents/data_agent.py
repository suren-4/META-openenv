# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Data Agent

"""
Conservative baseline agent.
Careful bidding with safety margins.
"""

try:
    from my_env.models import ArenaObservation, ArenaAction, ActionType
except ImportError:
    try:
        from ..models import ArenaObservation, ArenaAction, ActionType
    except (ImportError, ValueError):
        from models import ArenaObservation, ArenaAction, ActionType
from .base_agent import BaseAgent


class DataAgent(BaseAgent):
    """Conservative agent strategy."""

    def act(self, observation: ArenaObservation) -> ArenaAction:
        my_tasks = [t for t in observation.unassigned_tasks if t.primary_team == self.team]
        
        if my_tasks:
            target = my_tasks[0]
            
            # Conservative bid: inflated ETA and lower confidence
            return ArenaAction(
                action_type=ActionType.BID,
                agent_id=self.agent_id,
                task_id=target.id,
                resource_request=target.resources_required,
                price_offered=target.base_value,
                eta_minutes=int(target.deadline_minutes * 0.8), # promise earlier to win, but request high value
                confidence=0.6
            )
            
        return ArenaAction(action_type=ActionType.PASS, agent_id=self.agent_id)
