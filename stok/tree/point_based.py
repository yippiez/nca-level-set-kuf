
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Union, Tuple
from pydantic import BaseModel, model_validator

from stok.tree.experiment import ExperimentBase

import numpy as np


# Strategies
class PointSampleStrategy(ABC):
    @abstractmethod
    def __init__(self, n: int, bound_begin: np.ndarray, bound_end: np.ndarray) -> None:
        pass

    @abstractmethod
    def sample(self, sdf: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pass

class RandomSampleStrategy(PointSampleStrategy):
    def __init__(self, n: int, bound_begin: np.ndarray, bound_end: np.ndarray) -> None:
        self.n = n
        self.bound_begin = bound_begin
        self.bound_end = bound_end
        
    def sample(self, sdf: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        sample = np.random.uniform(self.bound_begin, self.bound_end, (self.n, 3))

        distances = sdf(sample)
        distances = distances.reshape(-1, 1)

        return sample, distances


# Experiment
class PointBasedExperimentResult(BaseModel):
    points: np.ndarray
    distances: np.ndarray
    
    @model_validator(mode='after')
    def validate_shapes(self):
        if not (self.points.ndim == 2 and self.points.shape[1] == 3):
            raise ValueError(f"Points must be a 2D array with shape (n, 3), got {self.points.shape}")
        
        if not (self.distances.ndim == 2 and self.distances.shape[1] == 1):
            raise ValueError(f"Distances must be a 2D array with shape (n, 1), got {self.distances.shape}")
        
        if self.points.shape[0] != self.distances.shape[0]:
            raise ValueError(f"Points and distances must have the same number of samples, "
                             f"got {self.points.shape[0]} and {self.distances.shape[0]}")
        
        return self
    
    class Config:
        arbitrary_types_allowed = True

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
        """Perform the experiment."""
        points, distances = self.sample_strategy.sample(self.sdf)
        return PointBasedExperimentResult(points=points, distances=distances)


if __name__ == "__main__":
    # Example usage
    def example_sdf(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points, axis=1) - 1.0  # Example SDF (sphere of radius 1)
    
    bound_begin = np.array([-2.0, -2.0, -2.0])
    bound_end = np.array([2.0, 2.0, 2.0])
    
    sample_strategy = RandomSampleStrategy(n=1000, bound_begin=bound_begin, bound_end=bound_end)
    
    experiment = PointBasedExperiment(sample_strategy=sample_strategy, sdf=example_sdf, 
                                       bound_begin=bound_begin, bound_end=bound_end)
    
    print("Dump Path:")
    print(experiment.dump_path)
    
    print()

    print("Doing Experiment:")
    result = experiment.do()
    print(result)
