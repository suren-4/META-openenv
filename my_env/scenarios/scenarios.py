# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Scenarios

"""
Pre-built demo scenarios for storytelling.
"""

def get_learning_arc_scenario():
    """Agents start greedy (Phase 1) and learn to cooperate (Phase 2)."""
    return {
        "name": "Learning Arc",
        "description": "Showcases transition from individual to collective rewards.",
        "config": {"phase_1_episodes": 10, "phase_2_episodes": 10}
    }

def get_crisis_recovery_scenario():
    """Injects hardware failure and forces emergency re-planning."""
    return {
        "name": "Crisis Recovery",
        "description": "25% CPU goes offline. Market surges.",
        "config": {"scenario": "failure"}
    }

def get_alliance_formation_scenario():
    """Forces ML agent and Data agent to cooperate on heavy tasks."""
    return {
        "name": "Alliance Formation",
        "description": "Only large compound tasks arrive.",
        "config": {"task_mix": "heavy"}
    }
