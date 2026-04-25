# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Task System

"""
Task generation using Poisson process with diverse resource profiles.
Supports 6 task types, subtask decomposition, and dependency chains.
"""

import random
import math
from typing import List, Optional
from uuid import uuid4

from ..models import Task, TaskType, Subtask, ResourceProfile
from ..config import (
    TASK_TEMPLATES, TASK_ARRIVAL_RATE,
    PREDICTABLE_TASK_RATIO, URGENT_TASK_TYPES,
)


class TaskGenerator:
    """Generates tasks via Poisson process with configurable arrival rate."""

    def __init__(self, arrival_rate: float = TASK_ARRIVAL_RATE, seed: Optional[int] = None):
        self.arrival_rate = arrival_rate
        self.rng = random.Random(seed)
        self._task_counter = 0

    def generate(self, current_step: int) -> List[Task]:
        """Generate new tasks for this step via Poisson process.

        Args:
            current_step: Current simulation step number.

        Returns:
            List of new Task objects.
        """
        # Poisson-distributed number of tasks
        num_tasks = self._poisson_sample(self.arrival_rate)
        tasks = []

        for _ in range(num_tasks):
            task = self._create_task(current_step)
            tasks.append(task)

        return tasks

    def _poisson_sample(self, lam: float) -> int:
        """Sample from Poisson distribution using inverse transform."""
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1

    def _create_task(self, current_step: int) -> Task:
        """Create a single task from templates."""
        self._task_counter += 1

        # Decide if predictable or urgent
        is_urgent = self.rng.random() > PREDICTABLE_TASK_RATIO

        if is_urgent:
            # Pick from urgent task types
            template = self.rng.choice(
                [t for t in TASK_TEMPLATES if t.task_type in URGENT_TASK_TYPES]
            )
        else:
            template = self.rng.choice(TASK_TEMPLATES)

        # Add some variance to resource requirements (±20%)
        variance = lambda v: max(1, int(v * self.rng.uniform(0.8, 1.2)))

        resources = ResourceProfile(
            cpu=variance(template.cpu),
            ram_gb=variance(template.ram_gb),
            gpu=template.gpu,  # GPU counts are discrete, keep exact
            bandwidth_mbps=variance(template.bandwidth_mbps),
        )

        # Add variance to deadline and value
        deadline = max(10, int(template.deadline_minutes * self.rng.uniform(0.8, 1.3)))
        value = round(template.base_value * self.rng.uniform(0.8, 1.2), 1)

        # Build subtasks
        subtasks = []
        for st_def in template.subtasks:
            subtasks.append(Subtask(
                name=st_def["name"],
                cpu=variance(st_def["cpu"]),
                ram_gb=variance(st_def["ram_gb"]),
                gpu=st_def.get("gpu", 0),
                bandwidth_mbps=variance(st_def.get("bandwidth_mbps", 0)),
                duration_minutes=max(5, int(st_def["duration_minutes"] * self.rng.uniform(0.8, 1.2))),
                value_share=st_def["value_share"],
                team=st_def["team"],
            ))

        task_id = f"T_{template.task_type[:4]}_{self._task_counter:04d}"

        return Task(
            id=task_id,
            task_type=TaskType(template.task_type),
            primary_team=template.primary_team,
            resources_required=resources,
            deadline_minutes=deadline,
            base_value=value,
            arrival_time=current_step,
            subtasks=subtasks,
        )

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = random.Random(seed)

    def reset(self):
        """Reset counter (keep seed)."""
        self._task_counter = 0


class TaskQueue:
    """Manages the global task queue — unassigned and in-progress tasks."""

    def __init__(self):
        self.unassigned: List[Task] = []
        self.in_progress: List[Task] = []
        self.completed: List[Task] = []
        self.failed: List[Task] = []

    def add_tasks(self, tasks: List[Task]):
        """Add new tasks to the unassigned queue."""
        self.unassigned.extend(tasks)

    def assign_task(self, task_id: str, agent_id: str, step: int) -> Optional[Task]:
        """Move a task from unassigned to in-progress."""
        for i, task in enumerate(self.unassigned):
            if task.id == task_id:
                task.assigned_to = agent_id
                task.started_at = step
                self.in_progress.append(task)
                self.unassigned.pop(i)
                return task
        return None

    def complete_task(self, task_id: str, step: int) -> Optional[Task]:
        """Move a task from in-progress to completed."""
        for i, task in enumerate(self.in_progress):
            if task.id == task_id:
                task.completed_at = step
                self.completed.append(task)
                self.in_progress.pop(i)
                return task
        return None

    def fail_task(self, task_id: str) -> Optional[Task]:
        """Move a task from in-progress to failed."""
        for i, task in enumerate(self.in_progress):
            if task.id == task_id:
                task.failed = True
                self.failed.append(task)
                self.in_progress.pop(i)
                return task
        return None

    def get_expired_tasks(self, current_step: int, minutes_per_step: int = 10) -> List[Task]:
        """Find tasks that have exceeded their deadline."""
        expired = []
        for task in self.in_progress:
            if task.started_at is not None:
                elapsed = (current_step - task.started_at) * minutes_per_step
                if elapsed > task.deadline_minutes:
                    expired.append(task)
        return expired

    def reset(self):
        """Clear all queues."""
        self.unassigned.clear()
        self.in_progress.clear()
        self.completed.clear()
        self.failed.clear()
