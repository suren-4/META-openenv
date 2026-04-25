# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Data Models

"""
Data models for the Cloud Resource Negotiation Arena.

Defines Actions, Observations, Tasks, and Resources using OpenEnv's
base types (Action, Observation) with Pydantic validation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


# =============================================================================
# Enums
# =============================================================================

class TaskType(str, Enum):
    API_DEPLOYMENT = "api_deployment"
    MODEL_TRAINING = "model_training"
    ETL_PIPELINE = "etl_pipeline"
    MONITORING_STACK = "monitoring_stack"
    AD_HOC_ANALYSIS = "ad_hoc_analysis"
    EMERGENCY_PATCH = "emergency_patch"


class AgentRole(str, Enum):
    FRONTEND = "frontend"
    ML_PIPELINE = "ml_pipeline"
    DATA_WAREHOUSE = "data_warehouse"
    DEVOPS = "devops"


class ActionType(str, Enum):
    """Discriminator for the unified ArenaAction."""
    BID = "bid"
    PROPOSE_COALITION = "propose_coalition"
    RESPOND_TO_PROPOSAL = "respond"
    RENEGOTIATE = "renegotiate"
    PASS = "pass"


class AdversarialEventType(str, Enum):
    ROGUE_AGENT = "rogue"
    RESOURCE_FAILURE = "failure"
    MARKET_MANIPULATION = "manipulation"
    INSIDER_LEAK = "leak"


# =============================================================================
# Resource Profile
# =============================================================================

class ResourceProfile(BaseModel):
    """Resource requirements or allocation for a task/cluster."""
    cpu: int = Field(default=0, ge=0, description="CPU cores")
    ram_gb: int = Field(default=0, ge=0, description="RAM in GB")
    gpu: int = Field(default=0, ge=0, description="GPU count")
    bandwidth_mbps: int = Field(default=0, ge=0, description="Bandwidth in Mbps")

    def fits_within(self, capacity: "ResourceProfile") -> bool:
        """Check if this profile fits within the given capacity."""
        return (
            self.cpu <= capacity.cpu
            and self.ram_gb <= capacity.ram_gb
            and self.gpu <= capacity.gpu
            and self.bandwidth_mbps <= capacity.bandwidth_mbps
        )

    def __add__(self, other: "ResourceProfile") -> "ResourceProfile":
        return ResourceProfile(
            cpu=self.cpu + other.cpu,
            ram_gb=self.ram_gb + other.ram_gb,
            gpu=self.gpu + other.gpu,
            bandwidth_mbps=self.bandwidth_mbps + other.bandwidth_mbps,
        )

    def __sub__(self, other: "ResourceProfile") -> "ResourceProfile":
        return ResourceProfile(
            cpu=max(0, self.cpu - other.cpu),
            ram_gb=max(0, self.ram_gb - other.ram_gb),
            gpu=max(0, self.gpu - other.gpu),
            bandwidth_mbps=max(0, self.bandwidth_mbps - other.bandwidth_mbps),
        )


# =============================================================================
# Task Models
# =============================================================================

class Subtask(BaseModel):
    """A decomposable piece of a larger task."""
    name: str
    cpu: int = 0
    ram_gb: int = 0
    gpu: int = 0
    bandwidth_mbps: int = 0
    duration_minutes: int = 60
    value_share: float = 0.0  # fraction of parent task value
    team: str = ""  # recommended team

    @property
    def resources(self) -> ResourceProfile:
        return ResourceProfile(
            cpu=self.cpu, ram_gb=self.ram_gb,
            gpu=self.gpu, bandwidth_mbps=self.bandwidth_mbps,
        )


class Task(BaseModel):
    """A work item in the task queue."""
    id: str
    task_type: TaskType
    primary_team: str
    resources_required: ResourceProfile
    deadline_minutes: int
    base_value: float
    arrival_time: int  # step number when task arrived
    subtasks: List[Subtask] = Field(default_factory=list)

    # Dependency fields
    parent_task_id: Optional[str] = None     # must complete first
    unlocks: List[str] = Field(default_factory=list)  # task IDs this enables
    cascade_penalty: float = 0.0             # penalty if parent fails

    # Runtime state
    assigned_to: Optional[str] = None        # agent_id or coalition_id
    started_at: Optional[int] = None         # step when execution began
    completed_at: Optional[int] = None
    failed: bool = False


class TaskCompletion(BaseModel):
    """Record of a completed task."""
    task_id: str
    task_type: TaskType
    assigned_agent: str
    coalition_members: List[str] = Field(default_factory=list)
    value_earned: float
    on_time: bool
    minutes_late: int = 0
    actual_resources_used: ResourceProfile = Field(default_factory=ResourceProfile)


class TaskProgress(BaseModel):
    """Progress of an in-flight task."""
    task_id: str
    task_type: TaskType
    progress_pct: float = 0.0  # 0–100
    time_remaining_minutes: int = 0
    resources_allocated: ResourceProfile = Field(default_factory=ResourceProfile)


# =============================================================================
# Agent Status Models
# =============================================================================

class AgentStatus(BaseModel):
    """Public information about an agent visible to all."""
    agent_id: str
    role: AgentRole
    reputation: float = 1.0
    tasks_in_flight_count: int = 0
    resource_reservations: ResourceProfile = Field(default_factory=ResourceProfile)
    cumulative_reward: float = 0.0
    is_online: bool = True


class MarketState(BaseModel):
    """Current market conditions."""
    demand_multiplier: float = 1.0  # 1.0x–3.0x surge pricing
    base_price_per_cpu: float = 1.0
    utilization_pct: float = 0.0    # overall cluster utilization


# =============================================================================
# Actions — Separate classes per action type
# =============================================================================

class ArenaAction(Action):
    """Unified action class for all agent actions."""
    action_type: ActionType = Field(default=ActionType.PASS, description="Action type discriminator")
    agent_id: str = Field(..., description="ID of the agent taking this action")
    
    # BID fields
    task_id: Optional[str] = Field(default=None, description="Task ID for bid/coalition")
    resource_request: Optional[ResourceProfile] = None
    price_offered: Optional[float] = None
    eta_minutes: Optional[int] = None
    confidence: Optional[float] = None
    
    # COALITION fields
    peer_agents: Optional[List[str]] = None
    subtask_split: Optional[Dict[str, str]] = None
    reward_split: Optional[Dict[str, float]] = None
    
    # RESPOND fields
    proposal_id: Optional[str] = None
    accept: Optional[bool] = None
    
    # RENEGOTIATE fields
    new_deadline: Optional[int] = None
    new_resource_split: Optional[Dict[str, ResourceProfile]] = None



# =============================================================================
# Proposals
# =============================================================================

class CoalitionProposal(BaseModel):
    """A pending coalition proposal."""
    proposal_id: str
    proposer: str
    task_id: str
    peer_agents: List[str]
    subtask_split: Dict[str, str]
    reward_split: Dict[str, float]
    votes: Dict[str, Optional[bool]] = Field(default_factory=dict)  # agent → accept/reject/None
    created_at: int = 0  # step number


# =============================================================================
# Observation
# =============================================================================

class ArenaObservation(Observation):
    """Per-agent observation of the marketplace.

    Contains both public state (visible to all) and private state
    (unique to the observing agent).
    """
    # Round info
    current_step: int = 0
    current_round: int = 0
    phase: int = 1  # 1 = individual, 2 = collective
    observing_agent: str = ""  # which agent this observation is for

    # Public state
    unassigned_tasks: List[Task] = Field(default_factory=list)
    cluster_utilization: ResourceProfile = Field(default_factory=ResourceProfile)
    utilization_forecast: List[ResourceProfile] = Field(default_factory=list)  # next 3 windows
    agent_roster: List[AgentStatus] = Field(default_factory=list)
    recent_completions: List[TaskCompletion] = Field(default_factory=list)
    pending_proposals: List[CoalitionProposal] = Field(default_factory=list)
    market_conditions: MarketState = Field(default_factory=MarketState)

    # Private state (per-agent)
    my_tasks_in_flight: List[TaskProgress] = Field(default_factory=list)
    my_resource_budget: ResourceProfile = Field(default_factory=ResourceProfile)
    trust_scores: Dict[str, float] = Field(default_factory=dict)

    # Adversarial / oversight info
    active_alerts: List[str] = Field(default_factory=list)
    oversight_report: Optional[str] = None

    # Negotiation log (recent actions by all agents)
    negotiation_log: List[Dict[str, Any]] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Render observation as natural language prompt for LLM agents.

        This is the KEY bridge between the environment and TRL training.
        Formats all state information into structured text that an LLM
        can reason over and produce action JSON.
        """
        lines = []
        lines.append(f"You are the {self.observing_agent} agent. Round {self.current_round}, "
                      f"Step {self.current_step}. Phase {self.phase}.")
        lines.append("")

        # Cluster state
        cap = self.cluster_utilization
        lines.append(f"CLUSTER STATUS: CPU {cap.cpu} used | RAM {cap.ram_gb}GB used | "
                      f"GPU {cap.gpu} used | BW {cap.bandwidth_mbps}Mbps used")
        lines.append(f"MARKET: Demand multiplier {self.market_conditions.demand_multiplier:.1f}x, "
                      f"Price/CPU {self.market_conditions.base_price_per_cpu:.1f}")
        lines.append("")

        # My state
        budget = self.my_resource_budget
        lines.append(f"YOUR BUDGET: CPU {budget.cpu} | RAM {budget.ram_gb}GB | "
                      f"GPU {budget.gpu} | BW {budget.bandwidth_mbps}Mbps")
        lines.append(f"YOUR TASKS IN FLIGHT: {len(self.my_tasks_in_flight)}")
        lines.append("")

        # Task queue
        lines.append("AVAILABLE TASKS:")
        if not self.unassigned_tasks:
            lines.append("  (none)")
        for t in self.unassigned_tasks[:10]:  # limit to avoid huge prompts
            affinity = " (YOUR SPECIALTY)" if t.primary_team == self.observing_agent else ""
            lines.append(
                f"  [{t.id}] {t.task_type.value} — "
                f"CPU:{t.resources_required.cpu} RAM:{t.resources_required.ram_gb}GB "
                f"GPU:{t.resources_required.gpu} — "
                f"deadline {t.deadline_minutes}min — value {t.base_value}"
                f"{affinity}"
            )
            if t.subtasks:
                lines.append(f"    Decomposable: {', '.join(s.name for s in t.subtasks)}")
        lines.append("")

        # Pending proposals
        if self.pending_proposals:
            lines.append("PENDING COALITION PROPOSALS:")
            for p in self.pending_proposals:
                lines.append(
                    f"  [{p.proposal_id}] {p.proposer} proposes coalition on {p.task_id}: "
                    f"split {p.reward_split}"
                )
            lines.append("")

        # Trust scores
        if self.trust_scores:
            trust_str = ", ".join(f"{k}={v:.2f}" for k, v in self.trust_scores.items())
            lines.append(f"TRUST SCORES: {trust_str}")
            lines.append("")

        # Recent completions
        if self.recent_completions:
            lines.append("RECENT COMPLETIONS:")
            for c in self.recent_completions[-5:]:
                status = "on-time" if c.on_time else f"{c.minutes_late}min late"
                lines.append(f"  {c.task_id} ({c.task_type.value}) — {status} — +{c.value_earned:.0f}")
            lines.append("")

        # Alerts
        if self.active_alerts:
            lines.append("⚠ ALERTS:")
            for alert in self.active_alerts:
                lines.append(f"  - {alert}")
            lines.append("")

        # Instructions
        lines.append("Choose one action: BID, PROPOSE_COALITION, RESPOND, RENEGOTIATE, or PASS.")
        lines.append("Respond with valid JSON for your chosen action type.")

        return "\n".join(lines)