# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Agent Models

"""
Internal agent state models — beliefs, budgets, learning model references.
These are server-side models (not sent over the wire).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .models import AgentRole, ResourceProfile, Task


class AgentInternalState(BaseModel):
    """Full internal state of an agent (server-side only)."""

    agent_id: str
    role: AgentRole
    reputation: float = 1.0

    # Resource budget for current episode
    resource_budget: ResourceProfile = Field(
        default_factory=lambda: ResourceProfile(cpu=125, ram_gb=512, gpu=2, bandwidth_mbps=25)
    )

    # Tasks currently being worked on
    tasks_in_flight: List[str] = Field(default_factory=list)  # task IDs

    # Reward tracking
    cumulative_reward: float = 0.0
    episode_reward: float = 0.0

    # Coalition history
    coalition_count: int = 0
    coalition_success_count: int = 0
    coalition_partners: Dict[str, int] = Field(default_factory=dict)  # partner → success count

    # Bid history
    bids_placed: int = 0
    bids_won: int = 0

    # Learning state references (not serialized — set in code)
    is_online: bool = True
    is_rogue: bool = False  # adversarial mode flag

    class Config:
        arbitrary_types_allowed = True

    @property
    def win_rate(self) -> float:
        if self.bids_placed == 0:
            return 0.0
        return self.bids_won / self.bids_placed

    @property
    def coalition_success_rate(self) -> float:
        if self.coalition_count == 0:
            return 0.0
        return self.coalition_success_count / self.coalition_count

    def reset_episode(self):
        """Reset per-episode state while preserving cross-episode metrics."""
        self.episode_reward = 0.0
        self.tasks_in_flight = []
        self.resource_budget = ResourceProfile(cpu=125, ram_gb=512, gpu=2, bandwidth_mbps=25)
        self.is_online = True
        self.is_rogue = False
