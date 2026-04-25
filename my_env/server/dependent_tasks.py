# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Dependent Task System

"""
Cascading task dependencies. Tasks can unlock other tasks,
and failing a parent task can incur cascade penalties.
"""

try:
    from my_env.models import Task, TaskCompletion
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import Task, TaskCompletion
    except (ImportError, ModuleNotFoundError):
        from models import Task, TaskCompletion

class TaskDependencyGraph:
    """Manages dependencies between tasks and checks for unlocks."""

    def __init__(self):
        self.unlocked_tasks: List[str] = []
        self.completed_tasks: set = set()

    def check_unlocks(self, new_completions: List[TaskCompletion]) -> List[str]:
        """
        Check if any new task completions unlock dependent tasks.
        (This is a simplified stub. In a full implementation, this would
        actually track a DAG of tasks and release new ones.)
        """
        unlocked_this_round = []
        for completion in new_completions:
            self.completed_tasks.add(completion.task_id)
            # In our simulation, tasks don't have hardcoded static unlocks yet,
            # but if they did we would add them here.

        return unlocked_this_round

    def reset(self):
        self.unlocked_tasks.clear()
        self.completed_tasks.clear()
