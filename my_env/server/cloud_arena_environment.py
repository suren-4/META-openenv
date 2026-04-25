# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Main Environment

"""
Core OpenEnv Environment for the Cloud Resource Negotiation Arena.

Multi-agent environment where 4 AI agents negotiate, bid, and form
coalitions over shared Kubernetes cluster resources.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from my_env.models import (
        ArenaObservation, ActionType, AgentRole, AgentStatus, TaskCompletion, TaskProgress,
        ResourceProfile, MarketState, CoalitionProposal, Task, ArenaAction
    )
    from my_env.agent_models import AgentInternalState
    from my_env.config import (
        AGENT_ROLES, EPISODE_LENGTH, TASK_ARRIVAL_RATE,
        NUM_AGENTS, PHASE_1_EPISODES,
        CLUSTER_CPU_TOTAL, CLUSTER_RAM_TOTAL_GB,
        CLUSTER_GPU_TOTAL, CLUSTER_BANDWIDTH_TOTAL_MBPS,
    )
    from my_env.server.cluster_state import ClusterState
    from my_env.server.task_system import TaskGenerator, TaskQueue
    from my_env.server.negotiation_engine import NegotiationEngine
    from my_env.server.reward_system import RewardCalculator
    from my_env.server.trust_manager import TrustManager
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import (
            ArenaObservation, ActionType, AgentRole, AgentStatus, TaskCompletion, TaskProgress,
            ResourceProfile, MarketState, CoalitionProposal, Task, ArenaAction
        )
        from ..agent_models import AgentInternalState
        from ..config import (
            AGENT_ROLES, EPISODE_LENGTH, TASK_ARRIVAL_RATE,
            NUM_AGENTS, PHASE_1_EPISODES,
            CLUSTER_CPU_TOTAL, CLUSTER_RAM_TOTAL_GB,
            CLUSTER_GPU_TOTAL, CLUSTER_BANDWIDTH_TOTAL_MBPS,
        )
        from .cluster_state import ClusterState
        from .task_system import TaskGenerator, TaskQueue
        from .negotiation_engine import NegotiationEngine
        from .reward_system import RewardCalculator
        from .trust_manager import TrustManager
    except (ImportError, ModuleNotFoundError):
        from models import (
            ArenaObservation, ActionType, AgentRole, AgentStatus, TaskCompletion, TaskProgress,
            ResourceProfile, MarketState, CoalitionProposal, Task, ArenaAction
        )
        from agent_models import AgentInternalState
        from config import (
            AGENT_ROLES, EPISODE_LENGTH, TASK_ARRIVAL_RATE,
            NUM_AGENTS, PHASE_1_EPISODES,
            CLUSTER_CPU_TOTAL, CLUSTER_RAM_TOTAL_GB,
            CLUSTER_GPU_TOTAL, CLUSTER_BANDWIDTH_TOTAL_MBPS,
        )
        from server.cluster_state import ClusterState
        from server.task_system import TaskGenerator, TaskQueue
        from server.negotiation_engine import NegotiationEngine
        from server.reward_system import RewardCalculator
        from server.trust_manager import TrustManager


class CloudArenaEnvironment(Environment):
    """
    Multi-agent Cloud Resource Negotiation Arena.

    Implements OpenEnv's Environment interface. Each step() call processes
    one agent's action. When all 4 agents have acted, the round resolves:
    bids assigned, coalitions formed, tasks progress, rewards computed.

    Lifecycle:
        reset() → step(agent1_action) → step(agent2_action) →
        step(agent3_action) → step(agent4_action) → [ROUND RESOLVES] →
        step(agent1_action) → ...
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        num_agents: int = NUM_AGENTS,
        episode_length: int = EPISODE_LENGTH,
        task_arrival_rate: float = TASK_ARRIVAL_RATE,
        scenario: str = "normal",
    ):
        super().__init__()
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.scenario = scenario

        # Core components
        capacity = ResourceProfile(
            cpu=CLUSTER_CPU_TOTAL, ram_gb=CLUSTER_RAM_TOTAL_GB,
            gpu=CLUSTER_GPU_TOTAL, bandwidth_mbps=CLUSTER_BANDWIDTH_TOTAL_MBPS,
        )
        self.cluster = ClusterState(capacity)
        self.task_gen = TaskGenerator(arrival_rate=task_arrival_rate)
        self.task_queue = TaskQueue()
        self.negotiation = NegotiationEngine()
        self.rewards_calc = RewardCalculator()
        from my_env.server.oversight_agent import OversightAgent
        self.oversight = OversightAgent()
        self.trust = TrustManager(agent_ids=AGENT_ROLES)

        # Agent states
        self.agents: Dict[str, AgentInternalState] = {}

        # Round management
        self._agent_order = list(AGENT_ROLES)
        self._current_agent_idx = 0
        self._actions_this_round: Dict[str, Any] = {}

        # Episode tracking
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_round = 0
        self._episode_num = 0
        self._recent_completions: List[TaskCompletion] = []
        self._negotiation_log: List[Dict[str, Any]] = []
        self._market_multiplier = 1.0

        # Metrics
        self._round_rewards: Dict[str, float] = {}

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ArenaObservation:
        """Reset the environment for a new episode."""
        if seed is not None:
            random.seed(seed)
            self.task_gen.set_seed(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._current_round = 0
        self._current_agent_idx = 0
        self._actions_this_round.clear()
        self._recent_completions.clear()
        self._negotiation_log.clear()
        self._round_rewards.clear()
        self._market_multiplier = 1.0
        self._episode_num += 1

        # Reset components
        self.cluster.reset()
        self.task_queue.reset()
        self.negotiation.reset()
        self.task_gen.reset()

        # Initialize agents
        self.agents = {
            role: AgentInternalState(agent_id=role, role=AgentRole(role))
            for role in AGENT_ROLES
        }

        # Generate initial tasks
        initial_tasks = self.task_gen.generate(0)
        self.task_queue.add_tasks(initial_tasks)

        # Build observation for the first agent
        first_agent = self._agent_order[0]
        return self._build_observation(first_agent)

    def step(self, action, timeout_s: Optional[float] = None, **kwargs) -> ArenaObservation:
        """Process one agent's action.

        When all agents have acted (4 actions buffered), resolves the round:
        assigns tasks, progresses time, computes rewards.
        """
        self._state.step_count += 1

        # Determine which agent is acting
        current_agent = self._agent_order[self._current_agent_idx]

        # Validate agent_id matches expected agent
        agent_id = getattr(action, 'agent_id', current_agent)

        # Log the action
        self._log_action(agent_id, action)

        # Buffer the action in the negotiation engine
        self.negotiation.buffer_action(action)
        self._actions_this_round[agent_id] = action

        # Advance to next agent
        self._current_agent_idx += 1

        # Check if all agents have acted → resolve round
        if self._current_agent_idx >= self.num_agents:
            self._resolve_round()
            self._current_agent_idx = 0
            self._current_round += 1

        # Determine next agent for observation
        next_agent = self._agent_order[self._current_agent_idx]

        # Check if episode is done
        done = self._current_round >= self.episode_length

        obs = self._build_observation(next_agent)
        obs.done = done

        # Set reward for the acting agent
        obs.reward = self._round_rewards.get(agent_id, 0.0)

        return obs

    def _resolve_round(self):
        """Resolve all buffered actions for this round."""
        current_step = self._current_round

        # 1. Generate new tasks
        new_tasks = self.task_gen.generate(current_step)
        self.task_queue.add_tasks(new_tasks)

        # 2. Snapshot cluster utilization for forecasting
        self.cluster.snapshot_utilization()

        # 3. Resolve bids and coalitions
        bid_results = self.negotiation.resolve_round(
            task_queue=self.task_queue,
            cluster=self.cluster,
            agents=self.agents,
            current_step=current_step,
        )

        # 4. Progress in-flight tasks — check completions and failures
        completions, failures = self._progress_tasks(current_step)
        self._recent_completions = completions

        # 5. Update trust based on completions
        for completion in completions:
            if completion.coalition_members:
                if completion.on_time:
                    self.trust.update_coalition_success(completion.coalition_members)
                else:
                    self.trust.update_coalition_failure(completion.coalition_members)

        # 6. Compute rewards
        phase = 1 if self._episode_num <= PHASE_1_EPISODES else 2
        utilization = self.cluster.utilization()
        self._round_rewards = self.rewards_calc.compute_round_rewards(
            agents=self.agents,
            completions=completions,
            failures=failures,
            cluster_utilization_pct=utilization.get("overall", 0),
            phase=phase,
            market_multiplier=self._market_multiplier,
        )

        # 7. Apply rewards to agent states
        for agent_id, reward in self._round_rewards.items():
            if agent_id in self.agents:
                self.agents[agent_id].cumulative_reward += reward
                self.agents[agent_id].episode_reward += reward

        # 8. Run Oversight Analysis
        if hasattr(self, 'oversight'):
            self.oversight.analyze(
                actions=self._actions_this_round,
                assignments=[],  # can be expanded
                completions=completions
            )

        # 9. Clear round actions
        self._actions_this_round.clear()

    def _progress_tasks(self, current_step: int) -> tuple:
        """Advance in-flight tasks and check for completions/failures.

        Returns:
            (completions, failure_task_ids)
        """
        completions: List[TaskCompletion] = []
        failures: List[str] = []
        minutes_per_step = 10

        # Check for expired tasks
        expired = self.task_queue.get_expired_tasks(current_step, minutes_per_step)
        for task in expired:
            elapsed = (current_step - (task.started_at or 0)) * minutes_per_step
            minutes_late = max(0, elapsed - task.deadline_minutes)

            if minutes_late > 60:
                # Task failed
                self.task_queue.fail_task(task.id)
                self.cluster.release(task.id)
                failures.append(task.id)
            else:
                # Task completed (possibly late)
                self.task_queue.complete_task(task.id, current_step)
                self.cluster.release(task.id)

                # Determine coalition members
                coalition_members = []
                coalition = self.negotiation.active_coalitions.get(task.id)
                if coalition:
                    coalition_members = [coalition.proposer] + coalition.peer_agents

                completions.append(TaskCompletion(
                    task_id=task.id,
                    task_type=task.task_type,
                    assigned_agent=task.assigned_to or "",
                    coalition_members=coalition_members,
                    value_earned=task.base_value,
                    on_time=(minutes_late == 0),
                    minutes_late=minutes_late,
                    actual_resources_used=task.resources_required,
                ))

                # Update agent coalition partner counts
                if coalition_members:
                    for i, a in enumerate(coalition_members):
                        for b in coalition_members[i + 1:]:
                            if a in self.agents:
                                self.agents[a].coalition_partners[b] = \
                                    self.agents[a].coalition_partners.get(b, 0) + 1
                            if b in self.agents:
                                self.agents[b].coalition_partners[a] = \
                                    self.agents[b].coalition_partners.get(a, 0) + 1
                    for m in coalition_members:
                        if m in self.agents:
                            self.agents[m].coalition_success_count += 1

        # Also check for tasks that complete "naturally" (duration elapsed)
        for task in list(self.task_queue.in_progress):
            if task.started_at is not None:
                elapsed = (current_step - task.started_at) * minutes_per_step
                # Estimate: tasks complete when ~80% of deadline elapsed
                estimated_duration = task.deadline_minutes * 0.7
                if elapsed >= estimated_duration and task.id not in failures:
                    self.task_queue.complete_task(task.id, current_step)
                    self.cluster.release(task.id)

                    coalition_members = []
                    coalition = self.negotiation.active_coalitions.get(task.id)
                    if coalition:
                        coalition_members = [coalition.proposer] + coalition.peer_agents

                    minutes_late = max(0, int(elapsed - task.deadline_minutes))
                    completions.append(TaskCompletion(
                        task_id=task.id,
                        task_type=task.task_type,
                        assigned_agent=task.assigned_to or "",
                        coalition_members=coalition_members,
                        value_earned=task.base_value,
                        on_time=(minutes_late <= 0),
                        minutes_late=minutes_late,
                        actual_resources_used=task.resources_required,
                    ))

        # Remove completed task IDs from agent in-flight lists
        completed_ids = {c.task_id for c in completions} | set(failures)
        for agent in self.agents.values():
            agent.tasks_in_flight = [
                t for t in agent.tasks_in_flight if t not in completed_ids
            ]

        return completions, failures

    def _build_observation(self, agent_id: str) -> ArenaObservation:
        """Build observation for a specific agent."""
        agent = self.agents.get(agent_id)
        phase = 1 if self._episode_num <= PHASE_1_EPISODES else 2
        utilization = self.cluster.utilization()

        # Agent's in-flight task progress
        my_tasks = []
        if agent:
            for task_id in agent.tasks_in_flight:
                for task in self.task_queue.in_progress:
                    if task.id == task_id:
                        elapsed = (self._current_round - (task.started_at or 0)) * 10
                        progress = min(100, (elapsed / max(1, task.deadline_minutes)) * 100)
                        my_tasks.append(TaskProgress(
                            task_id=task.id,
                            task_type=task.task_type,
                            progress_pct=progress,
                            time_remaining_minutes=max(0, task.deadline_minutes - elapsed),
                            resources_allocated=task.resources_required,
                        ))

        # Build agent roster
        roster = []
        for aid, a in self.agents.items():
            roster.append(AgentStatus(
                agent_id=aid,
                role=a.role,
                reputation=a.reputation,
                tasks_in_flight_count=len(a.tasks_in_flight),
                resource_reservations=a.resource_budget,
                cumulative_reward=a.cumulative_reward,
                is_online=a.is_online,
            ))

        return ArenaObservation(
            current_step=self._state.step_count,
            current_round=self._current_round,
            phase=phase,
            observing_agent=agent_id,
            unassigned_tasks=list(self.task_queue.unassigned),
            cluster_utilization=ResourceProfile(
                cpu=self.cluster.allocated.cpu,
                ram_gb=self.cluster.allocated.ram_gb,
                gpu=self.cluster.allocated.gpu,
                bandwidth_mbps=self.cluster.allocated.bandwidth_mbps,
            ),
            utilization_forecast=self.cluster.forecast(3),
            agent_roster=roster,
            recent_completions=self._recent_completions[-10:],
            pending_proposals=self.negotiation.get_pending_proposals(),
            market_conditions=MarketState(
                demand_multiplier=self._market_multiplier,
                base_price_per_cpu=1.0,
                utilization_pct=utilization.get("overall", 0),
            ),
            my_tasks_in_flight=my_tasks,
            my_resource_budget=agent.resource_budget if agent else ResourceProfile(),
            trust_scores=self.trust.get_agent_trust_scores(agent_id),
            active_alerts=[],
            oversight_report=None,
            negotiation_log=self._negotiation_log[-20:],
            done=False,
            reward=0.0,
        )

    def _log_action(self, agent_id: str, action):
        """Log an action for the negotiation log."""
        action_type = getattr(action, 'action_type', ActionType.PASS)
        log_entry = {
            "agent": agent_id,
            "action_type": action_type.value if hasattr(action_type, 'value') else str(action_type),
            "round": self._current_round,
            "step": self._state.step_count,
        }

        if action.action_type == ActionType.BID:
            log_entry["task_id"] = action.task_id
            log_entry["price"] = action.price_offered
            log_entry["eta"] = action.eta_minutes
        elif action.action_type == ActionType.PROPOSE_COALITION:
            log_entry["task_id"] = action.task_id
            log_entry["peers"] = action.peer_agents
            log_entry["reward_split"] = action.reward_split
        elif action.action_type == ActionType.RESPOND_TO_PROPOSAL:
            log_entry["proposal_id"] = action.proposal_id
            log_entry["accepted"] = action.accept

        self._negotiation_log.append(log_entry)

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state
