from abc import ABC, abstractmethod
import math
from typing import Dict

class DropRateScheduler(ABC):
    """Base class for drop rate schedulers."""
    
    @abstractmethod
    def get_drop_rate(self) -> float:
        """Returns the current drop rate."""
        pass

class LinearDropRateScheduler(DropRateScheduler):
    """Linear drop rate scheduler.

    Args:
        initial_drop_rate (float): The initial drop rate.
        final_drop_rate (float): The final drop rate.
        num_steps (int): The number of steps to reach the final drop rate.
    """
    def __init__(self, initial_drop_rate: float, final_drop_rate: float, num_steps: int):
        self.initial_drop_rate = initial_drop_rate
        self.final_drop_rate = final_drop_rate
        self.num_steps = num_steps
        self.step_count = 0

    def get_drop_rate(self) -> float:
        if self.step_count >= self.num_steps:
            return self.final_drop_rate
        return self.initial_drop_rate + (self.final_drop_rate - self.initial_drop_rate) * (self.step_count / self.num_steps)

    def step(self) -> None:
        self.step_count += 1

class CosineAnnealingDropRateScheduler(DropRateScheduler):
    """Cosine annealing drop rate scheduler.

    Args:
        initial_drop_rate (float): The initial drop rate.
        final_drop_rate (float): The final drop rate.
        num_steps (int): The number of steps to reach the final drop rate.
    """
    def __init__(self, initial_drop_rate: float, final_drop_rate: float, num_steps: int):
        self.initial_drop_rate = initial_drop_rate
        self.final_drop_rate = final_drop_rate
        self.num_steps = num_steps
        self.step_count = 0

    def get_drop_rate(self) -> float:
        if self.step_count >= self.num_steps:
            return self.final_drop_rate
        return self.final_drop_rate + 0.5 * (self.initial_drop_rate - self.final_drop_rate) * (1 + math.cos(math.pi * self.step_count / self.num_steps))

    def step(self) -> None:
        self.step_count += 1

class StepDropRateScheduler(DropRateScheduler):
    """Step drop rate scheduler.

    Args:
        initial_drop_rate (float): The initial drop rate.
        drop_rate_schedule (Dict[int, float]): A dictionary mapping step numbers to drop rates.
    """
    def __init__(self, initial_drop_rate: float, drop_rate_schedule: Dict[int, float]):
        self.initial_drop_rate = initial_drop_rate
        self.drop_rate_schedule = drop_rate_schedule
        self.step_count = 0

    def get_drop_rate(self) -> float:
        applicable_steps = [step for step in self.drop_rate_schedule if step <= self.step_count]
        if applicable_steps:
            closest_step = max(applicable_steps)
            return self.drop_rate_schedule[closest_step]
        else:
            return self.initial_drop_rate

    def step(self) -> None:
        self.step_count += 1