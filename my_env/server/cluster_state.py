# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Cluster State

"""
Simulates the shared Kubernetes cluster's resource pool.
Tracks allocation, utilization, and supports resource failures.
"""

from typing import Dict, List, Optional
from ..models import ResourceProfile
from ..config import (
    CLUSTER_CPU_TOTAL, CLUSTER_RAM_TOTAL_GB,
    CLUSTER_GPU_TOTAL, CLUSTER_BANDWIDTH_TOTAL_MBPS,
)


class ClusterState:
    """Manages the simulated K8s cluster resource pool."""

    def __init__(self, capacity: Optional[ResourceProfile] = None):
        self.capacity = capacity or ResourceProfile(
            cpu=CLUSTER_CPU_TOTAL,
            ram_gb=CLUSTER_RAM_TOTAL_GB,
            gpu=CLUSTER_GPU_TOTAL,
            bandwidth_mbps=CLUSTER_BANDWIDTH_TOTAL_MBPS,
        )
        self.allocated = ResourceProfile(cpu=0, ram_gb=0, gpu=0, bandwidth_mbps=0)

        # Track per-task allocations for release
        self._allocations: Dict[str, ResourceProfile] = {}

        # Utilization history for forecasting
        self._utilization_history: List[ResourceProfile] = []

        # Resource failures (temporary capacity reductions)
        self._failures: Dict[str, int] = {}  # resource_type → amount_offline

    @property
    def available(self) -> ResourceProfile:
        """Currently available resources (capacity minus allocated minus failures)."""
        effective_capacity = self._effective_capacity()
        return effective_capacity - self.allocated

    def _effective_capacity(self) -> ResourceProfile:
        """Capacity after accounting for failures."""
        return ResourceProfile(
            cpu=max(0, self.capacity.cpu - self._failures.get("cpu", 0)),
            ram_gb=max(0, self.capacity.ram_gb - self._failures.get("ram_gb", 0)),
            gpu=max(0, self.capacity.gpu - self._failures.get("gpu", 0)),
            bandwidth_mbps=max(0, self.capacity.bandwidth_mbps - self._failures.get("bandwidth_mbps", 0)),
        )

    def allocate(self, task_id: str, resources: ResourceProfile) -> bool:
        """Try to allocate resources for a task. Returns True if successful."""
        available = self.available
        if resources.fits_within(available):
            self.allocated = self.allocated + resources
            self._allocations[task_id] = resources
            return True
        return False

    def release(self, task_id: str) -> Optional[ResourceProfile]:
        """Release resources allocated to a task."""
        resources = self._allocations.pop(task_id, None)
        if resources:
            self.allocated = self.allocated - resources
        return resources

    def utilization(self) -> Dict[str, float]:
        """Get current utilization percentages."""
        eff = self._effective_capacity()
        def _pct(used: int, total: int) -> float:
            return (used / total * 100) if total > 0 else 0.0

        return {
            "cpu": _pct(self.allocated.cpu, eff.cpu),
            "ram": _pct(self.allocated.ram_gb, eff.ram_gb),
            "gpu": _pct(self.allocated.gpu, eff.gpu),
            "bandwidth": _pct(self.allocated.bandwidth_mbps, eff.bandwidth_mbps),
            "overall": _pct(
                self.allocated.cpu + self.allocated.gpu * 50,  # weighted
                eff.cpu + eff.gpu * 50,
            ),
        }

    def snapshot_utilization(self):
        """Record current utilization for forecasting."""
        self._utilization_history.append(ResourceProfile(
            cpu=self.allocated.cpu,
            ram_gb=self.allocated.ram_gb,
            gpu=self.allocated.gpu,
            bandwidth_mbps=self.allocated.bandwidth_mbps,
        ))
        # Keep last 10 snapshots
        if len(self._utilization_history) > 10:
            self._utilization_history = self._utilization_history[-10:]

    def forecast(self, windows: int = 3) -> List[ResourceProfile]:
        """Forecast utilization for next N windows (simple moving average)."""
        if not self._utilization_history:
            return [self.allocated] * windows

        # Simple: return last known utilization repeated
        last = self._utilization_history[-1]
        return [last] * windows

    def apply_failure(self, resource_type: str, amount: int):
        """Simulate a resource failure — reduce effective capacity."""
        self._failures[resource_type] = self._failures.get(resource_type, 0) + amount

    def recover_failure(self, resource_type: str, amount: int):
        """Recover from a resource failure."""
        current = self._failures.get(resource_type, 0)
        self._failures[resource_type] = max(0, current - amount)

    def clear_failures(self):
        """Clear all resource failures."""
        self._failures.clear()

    def reset(self):
        """Reset cluster to initial state."""
        self.allocated = ResourceProfile(cpu=0, ram_gb=0, gpu=0, bandwidth_mbps=0)
        self._allocations.clear()
        self._utilization_history.clear()
        self._failures.clear()
