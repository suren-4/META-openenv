# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Rogue Agent

"""
Adversarial agent for stress testing.
Defects, submits false bids, and drops coalitions.
"""

try:
    from my_env.models import ArenaObservation, ArenaAction, ActionType
except ImportError:
    try:
        from ..models import ArenaObservation, ArenaAction, ActionType
    except (ImportError, ValueError):
        from models import ArenaObservation, ArenaAction, ActionType
from .base_agent import BaseAgent


class RogueAgent(BaseAgent):
    """Adversarial agent strategy."""

    def act(self, observation: ArenaObservation) -> ArenaAction:
        # Bid aggressively on everything with impossible ETAs
        if observation.unassigned_tasks:
            target = observation.unassigned_tasks[0]
            
            # Deceptive bid: 1 minute ETA, low price (to guarantee win), heavy resources
            return ArenaAction(
                action_type=ActionType.BID,
                agent_id=self.agent_id,
                task_id=target.id,
                resource_request=target.resources_required,
                price_offered=target.base_value * 0.1,
                eta_minutes=1,
                confidence=1.0
            )
            
        return ArenaAction(action_type=ActionType.PASS, agent_id=self.agent_id)
