# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Configuration

"""
Hyperparameters and constants for the Cloud Resource Negotiation Arena.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# =============================================================================
# Cluster Configuration
# =============================================================================

CLUSTER_CPU_TOTAL = 500          # CPU cores
CLUSTER_RAM_TOTAL_GB = 2048      # 2TB RAM
CLUSTER_GPU_TOTAL = 8            # GPUs
CLUSTER_BANDWIDTH_TOTAL_MBPS = 100  # Bandwidth

# =============================================================================
# Episode Configuration
# =============================================================================

EPISODE_LENGTH = 50              # Steps per episode (each step = 10 min simulated)
NUM_AGENTS = 4
TASK_ARRIVAL_RATE = 1.5          # Poisson lambda per 10-min window
TOTAL_TRAINING_EPISODES = 100
PHASE_1_EPISODES = 50           # Individual reward phase
PHASE_2_EPISODES = 50           # Collective reward phase (episodes 51-100)

# =============================================================================
# Task Definitions
# =============================================================================

@dataclass
class TaskTemplate:
    task_type: str
    primary_team: str
    cpu: int
    ram_gb: int
    gpu: int
    bandwidth_mbps: int
    deadline_minutes: int
    base_value: float
    example: str
    subtasks: List[Dict] = field(default_factory=list)


TASK_TEMPLATES = [
    TaskTemplate(
        task_type="api_deployment",
        primary_team="frontend",
        cpu=4, ram_gb=2, gpu=0, bandwidth_mbps=5,
        deadline_minutes=120,
        base_value=100.0,
        example="Deploy new checkout flow",
        subtasks=[
            {"name": "build_image", "cpu": 2, "ram_gb": 1, "gpu": 0, "bandwidth_mbps": 2,
             "duration_minutes": 30, "value_share": 0.3, "team": "devops"},
            {"name": "deploy_service", "cpu": 2, "ram_gb": 1, "gpu": 0, "bandwidth_mbps": 3,
             "duration_minutes": 60, "value_share": 0.7, "team": "frontend"},
        ],
    ),
    TaskTemplate(
        task_type="model_training",
        primary_team="ml_pipeline",
        cpu=8, ram_gb=4, gpu=2, bandwidth_mbps=2,
        deadline_minutes=480,
        base_value=300.0,
        example="Train XGBoost on Q2 data",
        subtasks=[
            {"name": "data_preprocessing", "cpu": 2, "ram_gb": 2, "gpu": 0, "bandwidth_mbps": 1,
             "duration_minutes": 60, "value_share": 0.17, "team": "data_warehouse"},
            {"name": "model_training", "cpu": 8, "ram_gb": 4, "gpu": 2, "bandwidth_mbps": 1,
             "duration_minutes": 240, "value_share": 0.66, "team": "ml_pipeline"},
            {"name": "validation_logging", "cpu": 2, "ram_gb": 1, "gpu": 0, "bandwidth_mbps": 0,
             "duration_minutes": 60, "value_share": 0.17, "team": "devops"},
        ],
    ),
    TaskTemplate(
        task_type="etl_pipeline",
        primary_team="data_warehouse",
        cpu=12, ram_gb=8, gpu=0, bandwidth_mbps=10,
        deadline_minutes=240,
        base_value=250.0,
        example="Ingest daily vendor data",
        subtasks=[
            {"name": "extract_transform", "cpu": 8, "ram_gb": 6, "gpu": 0, "bandwidth_mbps": 8,
             "duration_minutes": 120, "value_share": 0.6, "team": "data_warehouse"},
            {"name": "load_validate", "cpu": 4, "ram_gb": 2, "gpu": 0, "bandwidth_mbps": 2,
             "duration_minutes": 60, "value_share": 0.4, "team": "devops"},
        ],
    ),
    TaskTemplate(
        task_type="monitoring_stack",
        primary_team="devops",
        cpu=2, ram_gb=1, gpu=0, bandwidth_mbps=2,
        deadline_minutes=60,
        base_value=50.0,
        example="Deploy Prometheus scraper",
        subtasks=[],
    ),
    TaskTemplate(
        task_type="ad_hoc_analysis",
        primary_team="data_warehouse",
        cpu=4, ram_gb=2, gpu=0, bandwidth_mbps=3,
        deadline_minutes=360,
        base_value=150.0,
        example="Customer segmentation query",
        subtasks=[
            {"name": "query_execution", "cpu": 4, "ram_gb": 2, "gpu": 0, "bandwidth_mbps": 3,
             "duration_minutes": 180, "value_share": 0.7, "team": "data_warehouse"},
            {"name": "results_dashboard", "cpu": 1, "ram_gb": 1, "gpu": 0, "bandwidth_mbps": 1,
             "duration_minutes": 60, "value_share": 0.3, "team": "frontend"},
        ],
    ),
    TaskTemplate(
        task_type="emergency_patch",
        primary_team="frontend",
        cpu=2, ram_gb=1, gpu=0, bandwidth_mbps=2,
        deadline_minutes=30,
        base_value=200.0,
        example="Security hotfix deployment",
        subtasks=[],
    ),
]

# Task arrival mix: 70% predictable, 30% urgent
PREDICTABLE_TASK_RATIO = 0.7
URGENT_TASK_TYPES = ["emergency_patch", "monitoring_stack"]

# =============================================================================
# Reward Configuration
# =============================================================================

# Completion multipliers
REWARD_ON_TIME_MULTIPLIER = 1.0
REWARD_LATE_1_30_MULTIPLIER = 0.8
REWARD_LATE_30_60_MULTIPLIER = 0.5
REWARD_LATE_60_PLUS_PENALTY = -50.0

# Overhead factors
OVERHEAD_SOLO = 1.0
OVERHEAD_COALITION = 0.85

# Relationship bonuses
RELATIONSHIP_BONUS_PER_SUCCESS = 10.0
RELATIONSHIP_MAX_BONUS_COALITIONS = 3
ABANDONMENT_PENALTY = -25.0

# Phase 2 collective reward blend
COLLECTIVE_REWARD_INDIVIDUAL_WEIGHT = 0.7
COLLECTIVE_REWARD_COLLECTIVE_WEIGHT = 0.3

# Collective reward components
FAILED_TASK_PENALTY = 100.0
ON_TIME_BONUS = 20.0
CLUSTER_EFFICIENCY_BONUS = 50.0

# =============================================================================
# Trust Configuration
# =============================================================================

TRUST_INITIAL = 0.5
TRUST_SUCCESS_DELTA = 0.1
TRUST_FAILURE_DELTA = -0.05
TRUST_HALF_LIFE_EPISODES = 20
TRUST_MIN = 0.0
TRUST_MAX = 1.0

# Reputation threshold — agents below this lose bid tiebreakers
REPUTATION_THRESHOLD = 0.7

# =============================================================================
# Market Dynamics Configuration
# =============================================================================

BASE_PRICE_PER_CPU = 1.0
DEMAND_MULTIPLIER_MIN = 1.0
DEMAND_MULTIPLIER_MAX = 3.0
SURGE_UTILIZATION_THRESHOLD = 0.8   # Surge starts at 80% utilization

# =============================================================================
# Adversarial Configuration
# =============================================================================

# Probability of adversarial events per episode
ADVERSARIAL_ROGUE_PROBABILITY = 0.1
ADVERSARIAL_FAILURE_PROBABILITY = 0.15
ADVERSARIAL_MANIPULATION_PROBABILITY = 0.05

# Resource failure: fraction of resource type knocked offline
RESOURCE_FAILURE_FRACTION = 0.25  # 25% of a resource type goes offline

# =============================================================================
# Scenario Presets
# =============================================================================

SCENARIO_NORMAL = {
    "base_utilization": 0.8,
    "task_arrival_rate": 1.5,
    "adversarial_enabled": False,
}

SCENARIO_PEAK = {
    "base_utilization": 0.95,
    "task_arrival_rate": 2.5,
    "adversarial_enabled": False,
}

SCENARIO_FAILURE = {
    "base_utilization": 0.8,
    "task_arrival_rate": 1.5,
    "adversarial_enabled": True,
}

# =============================================================================
# Agent Roles
# =============================================================================

AGENT_ROLES = ["frontend", "ml_pipeline", "data_warehouse", "devops"]

AGENT_DISPLAY_NAMES = {
    "frontend": "Frontend Team",
    "ml_pipeline": "ML Pipeline Team",
    "data_warehouse": "Data Warehouse Team",
    "devops": "DevOps Team",
}
