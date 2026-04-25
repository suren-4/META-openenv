# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Deception Detection

"""
Detects false bids and manipulation attempts.
Records unreliable behaviors.
"""

from typing import Dict, List
from ..models import ArenaAction, ActionType
from ..agent_models import AgentInternalState


class DeceptionDetector:
    """Validates actions for deceptive patterns."""

    def __init__(self):
        self.suspicious_bids = []

    def validate_bids(self, actions_by_agent: Dict, agents: Dict[str, AgentInternalState]):
        """Check for suspiciously low bids or impossible ETA promises."""
        for agent_id, action in actions_by_agent.items():
            if action.action_type == ActionType.BID:
                # E.g., offering to do a heavy ML training task in 1 minute
                if action.eta_minutes < 5 and action.resource_request.gpu > 0:
                    self.suspicious_bids.append((agent_id, action.task_id, "impossible_eta"))

    def reset(self):
        self.suspicious_bids.clear()
