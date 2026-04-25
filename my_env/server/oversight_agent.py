# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Oversight Agent

"""
Fleet AI Bonus: Oversight Agent.
Monitors the ecosystem and explains agent behavior.
"""

from typing import Dict, List
try:
    from my_env.models import TaskCompletion
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import TaskCompletion
    except (ImportError, ModuleNotFoundError):
        from models import TaskCompletion

class OversightAgent:
    """Generates natural language explanations of environment dynamics."""

    def __init__(self):
        self.reports: List[str] = []

    def analyze(self, actions: Dict, assignments: List, completions: List[TaskCompletion]):
        """Analyze round results and generate a report."""
        report = []
        if len(completions) > 0:
            report.append(f"Successfully completed {len(completions)} tasks this round.")
        
        # Analyze actions
        bids = [a for a in actions.values() if getattr(a, 'action_type', '') == 'bid']
        coalitions = [a for a in actions.values() if getattr(a, 'action_type', '') == 'propose_coalition']
        
        if coalitions:
            report.append("Agents are actively forming coalitions to share resources.")
        elif len(bids) > 1:
            report.append(f"High competition detected: {len(bids)} bids placed in the last round.")
        elif bids:
            report.append("Routine single-agent bidding observed.")
        else:
            report.append("Agents are passing or conserving resources.")
            
        if assignments:
            report.append("New tasks have been assigned and cluster resources allocated.")
            
        if report:
            self.reports.append(" ".join(report))
        
    def get_latest_report(self) -> str:
        if self.reports:
            return self.reports[-1]
        return "Monitoring cluster nominal. Waiting for active negotiations."

    def reset(self):
        self.reports.clear()
