# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Base Agent

"""
Abstract base class for Python baseline agents.
"""

from abc import ABC, abstractmethod
from typing import List

try:
    from my_env.models import ArenaObservation, ArenaAction, Task
    from my_env.learning.cost_estimator import CostEstimator
    from my_env.learning.trust_graph import TrustGraph
    from my_env.learning.value_model import ValueModel
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import ArenaObservation, ArenaAction, Task
        from ..learning.cost_estimator import CostEstimator
        from ..learning.trust_graph import TrustGraph
        from ..learning.value_model import ValueModel
    except (ImportError, ModuleNotFoundError):
        from models import ArenaObservation, ArenaAction, Task
        from learning.cost_estimator import CostEstimator
        from learning.trust_graph import TrustGraph
        from learning.value_model import ValueModel


class BaseAgent(ABC):
    """Abstract base for all team agents."""

    def __init__(self, agent_id: str, team: str):
        self.agent_id = agent_id
        self.team = team
        
        # Learning models
        self.cost_estimator = CostEstimator()
        self.trust_graph = TrustGraph()
        self.value_model = ValueModel()

    @abstractmethod
    def act(self, observation: ArenaObservation) -> ArenaAction:
        """Select an action based on the current observation."""
        pass

    def update_learning(self, completions: List[Task], failures: List[Task]):
        """Update internal models after tasks complete/fail."""
        # This will be implemented fully in learning agents
        pass
