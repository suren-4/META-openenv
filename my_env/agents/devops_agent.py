# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — DevOps Agent

"""
Supportive baseline agent.
Joins coalitions and handles monitoring tasks.
"""

try:
    from my_env.models import ArenaObservation, ArenaAction, ActionType
except ImportError:
    try:
        from ..models import ArenaObservation, ArenaAction, ActionType
    except (ImportError, ValueError):
        from models import ArenaObservation, ArenaAction, ActionType
from .base_agent import BaseAgent


class DevOpsAgent(BaseAgent):
    """Supportive agent strategy."""

    def act(self, observation: ArenaObservation) -> ArenaAction:
        # 1. Accept any pending coalition proposals
        if observation.pending_proposals:
            return ArenaAction(
                action_type=ActionType.RESPOND_TO_PROPOSAL,
                agent_id=self.agent_id,
                proposal_id=observation.pending_proposals[0].proposal_id,
                accept=True
            )

        # 2. Otherwise bid on DevOps tasks
        my_tasks = [t for t in observation.unassigned_tasks if t.primary_team == self.team]
        if my_tasks:
            target = my_tasks[0]
            return ArenaAction(
                action_type=ActionType.BID,
                agent_id=self.agent_id,
                task_id=target.id,
                resource_request=target.resources_required,
                price_offered=target.base_value * 0.8,
                eta_minutes=target.deadline_minutes,
                confidence=0.9
            )
            
        return ArenaAction(action_type=ActionType.PASS, agent_id=self.agent_id)
