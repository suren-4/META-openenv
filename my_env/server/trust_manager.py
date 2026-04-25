# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Trust Manager

"""
Trust graph between agents for coalition partner selection.
Trust decays over time with configurable half-life.
"""

import math
from typing import Dict, Optional

from ..config import (
    TRUST_INITIAL, TRUST_SUCCESS_DELTA, TRUST_FAILURE_DELTA,
    TRUST_HALF_LIFE_EPISODES, TRUST_MIN, TRUST_MAX,
    AGENT_ROLES,
)


class TrustManager:
    """Manages inter-agent trust scores with time decay."""

    def __init__(self, agent_ids: Optional[list] = None):
        self.agent_ids = agent_ids or AGENT_ROLES
        # trust_scores[A][B] = A's trust in B
        self.trust_scores: Dict[str, Dict[str, float]] = {}
        self._interaction_step: Dict[str, Dict[str, int]] = {}
        self.decay_factor = 0.5 ** (1.0 / TRUST_HALF_LIFE_EPISODES)
        self._current_episode = 0
        self.reset()

    def reset(self):
        """Reset all trust scores to initial values."""
        self.trust_scores = {
            a: {b: TRUST_INITIAL for b in self.agent_ids if b != a}
            for a in self.agent_ids
        }
        self._interaction_step = {
            a: {b: 0 for b in self.agent_ids if b != a}
            for a in self.agent_ids
        }
        self._current_episode = 0

    def get_trust(self, from_agent: str, to_agent: str) -> float:
        """Get trust score (with time decay applied)."""
        if from_agent not in self.trust_scores:
            return TRUST_INITIAL
        if to_agent not in self.trust_scores[from_agent]:
            return TRUST_INITIAL

        base = self.trust_scores[from_agent][to_agent]
        last_interaction = self._interaction_step.get(from_agent, {}).get(to_agent, 0)
        episodes_since = max(0, self._current_episode - last_interaction)

        # Apply exponential decay toward TRUST_INITIAL
        if episodes_since > 0:
            decayed = TRUST_INITIAL + (base - TRUST_INITIAL) * (self.decay_factor ** episodes_since)
            return max(TRUST_MIN, min(TRUST_MAX, decayed))

        return base

    def get_trust_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get full trust matrix with decay applied."""
        matrix = {}
        for a in self.agent_ids:
            matrix[a] = {}
            for b in self.agent_ids:
                if a == b:
                    matrix[a][b] = 1.0
                else:
                    matrix[a][b] = self.get_trust(a, b)
        return matrix

    def get_agent_trust_scores(self, agent_id: str) -> Dict[str, float]:
        """Get all trust scores for a specific agent (decayed)."""
        scores = {}
        for other in self.agent_ids:
            if other != agent_id:
                scores[other] = self.get_trust(agent_id, other)
        return scores

    def update_coalition_success(self, members: list):
        """Update trust after successful coalition — all pairs get +delta."""
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                self._update_trust(a, b, TRUST_SUCCESS_DELTA)
                self._update_trust(b, a, TRUST_SUCCESS_DELTA)

    def update_coalition_failure(self, members: list, blamed_agent: Optional[str] = None):
        """Update trust after failed coalition.

        If blamed_agent is set, only that agent loses trust.
        Otherwise all members lose trust with each other.
        """
        if blamed_agent:
            for other in members:
                if other != blamed_agent:
                    self._update_trust(other, blamed_agent, TRUST_FAILURE_DELTA)
        else:
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    self._update_trust(a, b, TRUST_FAILURE_DELTA)
                    self._update_trust(b, a, TRUST_FAILURE_DELTA)

    def _update_trust(self, from_agent: str, to_agent: str, delta: float):
        """Update raw trust score and record interaction time."""
        if from_agent not in self.trust_scores:
            self.trust_scores[from_agent] = {}
        if to_agent not in self.trust_scores[from_agent]:
            self.trust_scores[from_agent][to_agent] = TRUST_INITIAL

        new_score = self.trust_scores[from_agent][to_agent] + delta
        self.trust_scores[from_agent][to_agent] = max(TRUST_MIN, min(TRUST_MAX, new_score))

        # Record interaction time
        if from_agent not in self._interaction_step:
            self._interaction_step[from_agent] = {}
        self._interaction_step[from_agent][to_agent] = self._current_episode

    def advance_episode(self):
        """Call at the start of each new episode to advance decay clock."""
        self._current_episode += 1
