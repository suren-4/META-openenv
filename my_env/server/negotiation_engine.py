# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Negotiation Engine

"""
Resolves bids, coalition proposals, and renegotiations at the end of each round.
"""

from typing import Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from my_env.models import (
        ArenaAction, ActionType, CoalitionProposal, Task, ResourceProfile,
    )
    from my_env.agent_models import AgentInternalState
    from my_env.server.cluster_state import ClusterState
    from my_env.server.task_system import TaskQueue
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import (
            ArenaAction, ActionType, CoalitionProposal, Task, ResourceProfile,
        )
        from ..agent_models import AgentInternalState
        from ..server.cluster_state import ClusterState
        from ..server.task_system import TaskQueue
    except (ImportError, ModuleNotFoundError):
        from models import (
            ArenaAction, ActionType, CoalitionProposal, Task, ResourceProfile,
        )
        from agent_models import AgentInternalState
        from server.cluster_state import ClusterState
        from server.task_system import TaskQueue


class BidResult:
    """Result of bid resolution for a single task."""
    def __init__(self, task_id: str, winner: Optional[str] = None,
                 resource_request: Optional[ResourceProfile] = None,
                 price: float = 0.0, is_coalition: bool = False):
        self.task_id = task_id
        self.winner = winner
        self.resource_request = resource_request
        self.price = price
        self.is_coalition = is_coalition


class NegotiationEngine:
    """Handles bid resolution, coalition formation, and renegotiation."""

    def __init__(self):
        self.pending_proposals: Dict[str, CoalitionProposal] = {}
        self.active_coalitions: Dict[str, CoalitionProposal] = {}  # task_id → proposal
        self._round_bids: List[ArenaAction] = []
        self._round_responses: List[ArenaAction] = []
        self._round_renegotiations: List[ArenaAction] = []

    def buffer_action(self, action) -> None:
        """Buffer an action for end-of-round resolution."""
        if action.action_type == ActionType.BID:
            self._round_bids.append(action)
        elif action.action_type == ActionType.PROPOSE_COALITION:
            self._create_proposal(action)
        elif action.action_type == ActionType.RESPOND_TO_PROPOSAL:
            self._round_responses.append(action)
        elif action.action_type == ActionType.RENEGOTIATE:
            self._round_renegotiations.append(action)
        # PassAction — nothing to buffer

    def _create_proposal(self, action: ArenaAction) -> str:
        """Create a new coalition proposal from an action."""
        proposal_id = f"P_{uuid4().hex[:8]}"
        proposal = CoalitionProposal(
            proposal_id=proposal_id,
            proposer=action.agent_id,
            task_id=action.task_id,
            peer_agents=action.peer_agents,
            subtask_split=action.subtask_split,
            reward_split=action.reward_split,
            votes={agent: None for agent in action.peer_agents},
        )
        # Proposer auto-accepts
        proposal.votes[action.agent_id] = True
        self.pending_proposals[proposal_id] = proposal
        return proposal_id

    def resolve_round(
        self,
        task_queue: TaskQueue,
        cluster: ClusterState,
        agents: Dict[str, AgentInternalState],
        current_step: int,
    ) -> List[BidResult]:
        """Resolve all buffered actions for this round.

        Order:
          1. Process coalition responses (accept/reject votes)
          2. Resolve completed coalitions (unanimous → assign)
          3. Resolve solo bids (highest price wins, ties broken by reputation)
          4. Process renegotiations

        Returns:
            List of BidResult for each resolved task assignment.
        """
        results: List[BidResult] = []

        # 1. Process coalition responses
        for response in self._round_responses:
            proposal = self.pending_proposals.get(response.proposal_id)
            if proposal and response.agent_id in proposal.votes:
                proposal.votes[response.agent_id] = response.accept

        # 2. Resolve completed coalition proposals
        resolved_tasks = set()
        for pid, proposal in list(self.pending_proposals.items()):
            all_voted = all(v is not None for v in proposal.votes.values())
            if all_voted:
                all_accepted = all(v for v in proposal.votes.values())
                if all_accepted:
                    # Coalition accepted — try to assign task
                    task = self._find_task(task_queue, proposal.task_id)
                    if task:
                        # Calculate total resources needed
                        total_resources = task.resources_required
                        if cluster.allocate(task.id, total_resources):
                            task_queue.assign_task(task.id, f"coalition_{pid}", current_step)
                            self.active_coalitions[task.id] = proposal
                            resolved_tasks.add(task.id)
                            results.append(BidResult(
                                task_id=task.id,
                                winner=f"coalition_{pid}",
                                resource_request=total_resources,
                                price=task.base_value,
                                is_coalition=True,
                            ))

                            # Update agent states
                            all_members = [proposal.proposer] + proposal.peer_agents
                            for member in all_members:
                                if member in agents:
                                    agents[member].tasks_in_flight.append(task.id)
                                    agents[member].coalition_count += 1

                # Remove resolved proposal
                del self.pending_proposals[pid]

        # 3. Resolve solo bids — group by task_id, pick winner
        bids_by_task: Dict[str, List[ArenaAction]] = {}
        for bid in self._round_bids:
            if bid.task_id not in resolved_tasks:
                bids_by_task.setdefault(bid.task_id, []).append(bid)

        for task_id, bids in bids_by_task.items():
            task = self._find_task(task_queue, task_id)
            if not task:
                continue

            # Sort: highest price first, then by reputation (tiebreaker)
            bids.sort(key=lambda b: (
                b.price_offered if b.price_offered is not None else 0.0,
                agents.get(b.agent_id, AgentInternalState(agent_id="", role="frontend")).reputation,
            ), reverse=True)

            # Try bids in order until one can be allocated
            for bid in bids:
                req_resources = bid.resource_request if bid.resource_request is not None else task.resources_required
                if cluster.allocate(task_id, req_resources):
                    task_queue.assign_task(task_id, bid.agent_id, current_step)
                    results.append(BidResult(
                        task_id=task_id,
                        winner=bid.agent_id,
                        resource_request=req_resources,
                        price=bid.price_offered if bid.price_offered is not None else task.base_value,
                        is_coalition=False,
                    ))

                    # Update agent state
                    if bid.agent_id in agents:
                        agents[bid.agent_id].tasks_in_flight.append(task_id)
                        agents[bid.agent_id].bids_won += 1
                    break

            # Track bids placed for all bidders
            for bid in bids:
                if bid.agent_id in agents:
                    agents[bid.agent_id].bids_placed += 1

        # 4. Process renegotiations (simplified: update deadline if majority agree)
        for reneg in self._round_renegotiations:
            coalition = self.active_coalitions.get(reneg.task_id)
            if coalition and reneg.new_deadline:
                # For simplicity, auto-accept renegotiations for now
                task = self._find_in_progress(task_queue, reneg.task_id)
                if task:
                    task.deadline_minutes = reneg.new_deadline

        # Clear round buffers
        self._round_bids.clear()
        self._round_responses.clear()
        self._round_renegotiations.clear()

        return results

    def _find_task(self, task_queue: TaskQueue, task_id: str) -> Optional[Task]:
        """Find a task in the unassigned queue."""
        for task in task_queue.unassigned:
            if task.id == task_id:
                return task
        return None

    def _find_in_progress(self, task_queue: TaskQueue, task_id: str) -> Optional[Task]:
        """Find a task in the in-progress queue."""
        for task in task_queue.in_progress:
            if task.id == task_id:
                return task
        return None

    def get_pending_proposals(self) -> List[CoalitionProposal]:
        """Get all pending proposals."""
        return list(self.pending_proposals.values())

    def reset(self):
        """Reset all state."""
        self.pending_proposals.clear()
        self.active_coalitions.clear()
        self._round_bids.clear()
        self._round_responses.clear()
        self._round_renegotiations.clear()
