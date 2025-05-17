
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, ClassVar, Union, Tuple
from pydantic import BaseModel
from .experiment import ExperimentBase 

import numpy as np


# Strategies
class PointSampleStrategy(ABC):
    @abstractmethod
    def __init__(self, n: int, bound_begin: np.ndarray, bound_end: np.ndarray) -> None:
        pass

    @abstractmethod
    def sample(self, sdf: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        pass

class RandomSampleStrategy(PointSampleStrategy):
    def __init__(self, n: int, bound_begin: np.ndarray, bound_end: np.ndarray) -> None:
        self.n = n
        self.bound_begin = bound_begin
        self.bound_end = bound_end
        
    def sample(self, sdf: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        sample = np.random.uniform(self.bound_begin, self.bound_end, (self.n, 3))

        distances = sdf(sample)
        distances = distances.reshape(-1, 1)

        combined = np.hstack((sample, distances))

        assert combined.shape == (self.n, 4), f"The outputted shape from sampling is expected to be {(self.n, 4)} but is {combined.shape}"

        return combined

# Experiment

class PointBasedExperimentResult(BaseModel):
    ...

class PointBasedExperiment(ExperimentBase):
    NAME: ClassVar[str] = "PointBasedExperiment"
    VERSION: ClassVar[str] = "v1a"
    
    def __init__(self, 
                 sample_strategy: PointSampleStrategy,
                 sdf: Callable[[np.ndarray], np.ndarray], 
                 bound_begin: Union[np.ndarray, Tuple[float, float, float]], 
                 bound_end: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        super().__init__()
        self.sdf = sdf
        self.sample_strategy = sample_strategy
        self.bound_begin = bound_begin if isinstance(bound_begin, np.ndarray) else np.array(bound_begin)
        self.bound_end = bound_end if isinstance(bound_end, np.ndarray) else np.array(bound_end)
        
    def do(self) -> PointBasedExperimentResult:
        pass
