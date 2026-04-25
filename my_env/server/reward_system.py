# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Reward System

"""
Multi-layer reward computation: individual + collective + relationship.
Phase-aware: Phase 1 (episodes 1-50) individual only,
Phase 2 (episodes 51-100) blended 70/30 individual/collective.
"""

from typing import Dict, List

from ..models import Task, TaskCompletion, ResourceProfile
from ..agent_models import AgentInternalState
from ..config import (
    REWARD_ON_TIME_MULTIPLIER, REWARD_LATE_1_30_MULTIPLIER,
    REWARD_LATE_30_60_MULTIPLIER, REWARD_LATE_60_PLUS_PENALTY,
    OVERHEAD_SOLO, OVERHEAD_COALITION,
    RELATIONSHIP_BONUS_PER_SUCCESS, RELATIONSHIP_MAX_BONUS_COALITIONS,
    ABANDONMENT_PENALTY,
    COLLECTIVE_REWARD_INDIVIDUAL_WEIGHT, COLLECTIVE_REWARD_COLLECTIVE_WEIGHT,
    FAILED_TASK_PENALTY, ON_TIME_BONUS, CLUSTER_EFFICIENCY_BONUS,
)


class RewardCalculator:
    """Computes rewards for all agents each round."""

    def compute_completion_multiplier(self, minutes_late: int) -> float:
        """Get reward multiplier based on lateness."""
        if minutes_late <= 0:
            return REWARD_ON_TIME_MULTIPLIER
        elif minutes_late <= 30:
            return REWARD_LATE_1_30_MULTIPLIER
        elif minutes_late <= 60:
            return REWARD_LATE_30_60_MULTIPLIER
        else:
            return 0.0  # plus penalty applied separately

    def compute_individual_reward(
        self,
        agent: AgentInternalState,
        completions: List[TaskCompletion],
        failures: List[str],
        market_multiplier: float = 1.0,
    ) -> float:
        """Compute individual agent reward for this round.

        Formula:
            reward = (task_value × completion_multiplier)
                   - (resource_cost × overhead_factor)
                   + relationship_bonus
        """
        reward = 0.0

        for completion in completions:
            # Only count completions that involve this agent
            is_involved = (
                completion.assigned_agent == agent.agent_id
                or agent.agent_id in completion.coalition_members
            )
            if not is_involved:
                continue

            # Task value × completion multiplier
            multiplier = self.compute_completion_multiplier(completion.minutes_late)
            task_reward = completion.value_earned * multiplier

            # Resource cost
            res = completion.actual_resources_used
            resource_cost = (res.cpu * 0.5 + res.ram_gb * 0.1 + res.gpu * 5.0
                             + res.bandwidth_mbps * 0.2) * market_multiplier

            # Overhead factor
            is_coalition = len(completion.coalition_members) > 0
            overhead = OVERHEAD_COALITION if is_coalition else OVERHEAD_SOLO

            reward += task_reward - (resource_cost * overhead)

            # Relationship bonus for coalition success
            if is_coalition and completion.on_time:
                for partner in completion.coalition_members:
                    if partner != agent.agent_id:
                        partner_count = agent.coalition_partners.get(partner, 0)
                        if partner_count < RELATIONSHIP_MAX_BONUS_COALITIONS:
                            reward += RELATIONSHIP_BONUS_PER_SUCCESS

            # Late penalty
            if completion.minutes_late > 60:
                reward += REWARD_LATE_60_PLUS_PENALTY

        # Penalty for failed tasks this agent was responsible for
        for task_id in failures:
            if task_id in agent.tasks_in_flight:
                reward += ABANDONMENT_PENALTY

        return reward

    def compute_collective_reward(
        self,
        all_completions: List[TaskCompletion],
        all_failures: List[str],
        cluster_utilization_pct: float,
    ) -> float:
        """Compute collective reward for the entire marketplace.

        Formula:
            collective = Σ(completed_values) - Σ(resource_costs)
                       - (failed_tasks × 100)
                       + (on_time_bonus × early_completions)
                       + (efficiency_bonus × utilization_ratio)
        """
        total_value = sum(c.value_earned for c in all_completions)
        total_cost = sum(
            c.actual_resources_used.cpu * 0.5 + c.actual_resources_used.ram_gb * 0.1
            for c in all_completions
        )
        failure_cost = len(all_failures) * FAILED_TASK_PENALTY
        on_time_count = sum(1 for c in all_completions if c.on_time)
        on_time_reward = on_time_count * ON_TIME_BONUS
        efficiency_reward = CLUSTER_EFFICIENCY_BONUS * (cluster_utilization_pct / 100.0)

        return total_value - total_cost - failure_cost + on_time_reward + efficiency_reward

    def compute_round_rewards(
        self,
        agents: Dict[str, AgentInternalState],
        completions: List[TaskCompletion],
        failures: List[str],
        cluster_utilization_pct: float,
        phase: int,
        market_multiplier: float = 1.0,
    ) -> Dict[str, float]:
        """Compute rewards for all agents for one round.

        Args:
            agents: All agent states.
            completions: Tasks completed this round.
            failures: Task IDs that failed this round.
            cluster_utilization_pct: Current cluster utilization %.
            phase: 1 (individual) or 2 (individual+collective blend).
            market_multiplier: Current market demand multiplier.

        Returns:
            Dict mapping agent_id → reward for this round.
        """
        rewards = {}

        for agent_id, agent in agents.items():
            individual = self.compute_individual_reward(
                agent, completions, failures, market_multiplier
            )

            if phase == 2:
                collective = self.compute_collective_reward(
                    completions, failures, cluster_utilization_pct
                )
                blended = (
                    COLLECTIVE_REWARD_INDIVIDUAL_WEIGHT * individual
                    + COLLECTIVE_REWARD_COLLECTIVE_WEIGHT * collective
                )
                rewards[agent_id] = blended
            else:
                rewards[agent_id] = individual

        return rewards
