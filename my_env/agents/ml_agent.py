# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — ML Agent

"""
Coalition-focused baseline agent.
Often proposes coalitions for large model training tasks.
"""

try:
    from my_env.models import ArenaObservation, ArenaAction, ActionType
except ImportError:
    try:
        from ..models import ArenaObservation, ArenaAction, ActionType
    except (ImportError, ValueError):
        from models import ArenaObservation, ArenaAction, ActionType
from .base_agent import BaseAgent


class MLAgent(BaseAgent):
    """Coalition-focused agent strategy."""

    def act(self, observation: ArenaObservation) -> ArenaAction:
        my_tasks = [t for t in observation.unassigned_tasks if t.primary_team == self.team]
        
        if my_tasks:
            target = my_tasks[0]
            
            # If the task has subtasks and we have peers available, propose a coalition
            if target.subtasks and len(observation.agent_roster) > 1:
                # Find a peer
                peer = None
                for a in observation.agent_roster:
                    if a.agent_id != self.agent_id and a.is_online:
                        peer = a.agent_id
                        break
                        
                if peer:
                    return ArenaAction(
                        action_type=ActionType.PROPOSE_COALITION,
                        agent_id=self.agent_id,
                        task_id=target.id,
                        peer_agents=[peer],
                        subtask_split={self.agent_id: target.subtasks[0].name},
                        reward_split={self.agent_id: 0.6, peer: 0.4}
                    )
            
            # Fallback to solo bid
            return ArenaAction(
                action_type=ActionType.BID,
                agent_id=self.agent_id,
                task_id=target.id,
                resource_request=target.resources_required,
                price_offered=target.base_value * 0.9, # Bid slightly lower to win
                eta_minutes=target.deadline_minutes,
                confidence=0.8
            )
            
        return ArenaAction(action_type=ActionType.PASS, agent_id=self.agent_id)
