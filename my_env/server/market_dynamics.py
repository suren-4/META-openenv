# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Market Dynamics

"""
Calculates dynamic pricing based on cluster demand (Surge Pricing).
"""

from typing import Dict
from ..config import BASE_PRICE_PER_CPU, DEMAND_MULTIPLIER_MIN, DEMAND_MULTIPLIER_MAX, SURGE_UTILIZATION_THRESHOLD


class MarketDynamics:
    """Manages surge pricing and market conditions."""

    def __init__(self):
        self.demand_multiplier = DEMAND_MULTIPLIER_MIN
        self.base_price_per_cpu = BASE_PRICE_PER_CPU

    def update(self, utilization: Dict[str, float]):
        """Update market multiplier based on overall cluster utilization."""
        overall_util = utilization.get("overall", 0.0) / 100.0

        if overall_util > SURGE_UTILIZATION_THRESHOLD:
            # Scale multiplier up to MAX linearly between THRESHOLD and 1.0
            excess = (overall_util - SURGE_UTILIZATION_THRESHOLD) / (1.0 - SURGE_UTILIZATION_THRESHOLD)
            self.demand_multiplier = DEMAND_MULTIPLIER_MIN + excess * (DEMAND_MULTIPLIER_MAX - DEMAND_MULTIPLIER_MIN)
        else:
            self.demand_multiplier = DEMAND_MULTIPLIER_MIN

    def reset(self):
        self.demand_multiplier = DEMAND_MULTIPLIER_MIN
