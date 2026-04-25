# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Adversarial Engine

"""
Injects adversarial events into the simulation:
- Rogue agents (greedy/defecting)
- Resource failures
- Market manipulation
"""

import random
from typing import Dict

from ..models import AdversarialEventType
from ..agent_models import AgentInternalState
from ..config import (
    ADVERSARIAL_ROGUE_PROBABILITY, ADVERSARIAL_FAILURE_PROBABILITY,
    ADVERSARIAL_MANIPULATION_PROBABILITY, RESOURCE_FAILURE_FRACTION,
)


class AdversarialEngine:
    """Manages randomized adversarial scenarios for stress testing."""

    def __init__(self, scenario: str = "normal", seed: int = None):
        self.scenario = scenario
        self.rng = random.Random(seed)
        self.active_events = []

    def maybe_trigger(self, current_step: int, agents: Dict[str, AgentInternalState]):
        """Potentially trigger an adversarial event this step."""
        if self.scenario != "failure":
            return

        # Attempt to trigger rogue agent
        if self.rng.random() < (ADVERSARIAL_ROGUE_PROBABILITY / 50.0): # scaled per step
            agent_id = self.rng.choice(list(agents.keys()))
            agents[agent_id].is_rogue = True
            self.active_events.append((current_step, AdversarialEventType.ROGUE_AGENT, agent_id))

        # Attempt to trigger resource failure
        # In actual implementation, we would callback to cluster state
        if self.rng.random() < (ADVERSARIAL_FAILURE_PROBABILITY / 50.0):
            res_type = self.rng.choice(["cpu", "ram_gb", "gpu"])
            self.active_events.append((current_step, AdversarialEventType.RESOURCE_FAILURE, res_type))

    def reset(self):
        self.active_events.clear()
