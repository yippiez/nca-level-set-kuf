
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Union, Tuple, List
from pydantic import BaseModel, model_validator

from stok.tree.experiment import ExperimentBase

import numpy as np


# Strategies
class PointSampleStrategy(ABC):
    @abstractmethod
    def __init__(self, n: int, bound_begin: np.ndarray, bound_end: np.ndarray) -> None:
        pass

    @abstractmethod
    def sample(self, sdfs: List[Callable[[np.ndarray], np.ndarray]], sdf_shape_values: List[float] = None) -> tuple[np.ndarray, np.ndarray]:
        pass

class RandomSampleStrategy(PointSampleStrategy):
    """Strategy for sampling 4D points (xyz + shape parameter)."""
    
    def __init__(self, n: int, bound_begin: np.ndarray, bound_end: np.ndarray, shape_values: List[float] = None) -> None:
        self.n = n
        self.bound_begin = bound_begin
        self.bound_end = bound_end
        self.shape_values = shape_values
        
    def sample(self, sdfs: List[Callable[[np.ndarray], np.ndarray]], sdf_shape_values: List[float] = None) -> tuple[np.ndarray, np.ndarray]:
        # Validate shape values are provided
        if sdf_shape_values is None or len(sdf_shape_values) < 1:
            raise ValueError("Shape values must be provided for 4D sampling")
            
        # Validate number of SDFs and shape values match
        if len(sdfs) != len(sdf_shape_values):
            raise ValueError(f"Number of SDFs ({len(sdfs)}) must match number of shape values ({len(sdf_shape_values)})")
        
        # Sample points with specific shape values
        # Distribute points evenly across shape values
        points_per_shape = self.n // len(sdf_shape_values)
        remaining_points = self.n % len(sdf_shape_values)
        
        samples = []
        all_distances = []
        
        for i, (sdf, shape_val) in enumerate(zip(sdfs, sdf_shape_values)):
            # Add extra point to first few shapes if n doesn't divide evenly
            n_points = points_per_shape + (1 if i < remaining_points else 0)
            
            # Sample 3D points
            xyz = np.random.uniform(
                self.bound_begin[:3], 
                self.bound_end[:3], 
                (n_points, 3)
            )
            
            # Get distances for these 3D points using the current SDF
            distances = sdf(xyz).reshape(-1, 1)
            
            # Add shape parameter to create 4D points
            shape_param = np.full((n_points, 1), shape_val)
            points_4d = np.concatenate([xyz, shape_param], axis=1)
            
            samples.append(points_4d)
            all_distances.append(distances)
        
        # Combine all samples and distances
        sample = np.vstack(samples)
        distances = np.vstack(all_distances)
        
        # Shuffle to mix shape values (keeping points and distances in sync)
        indices = np.arange(len(sample))
        np.random.shuffle(indices)
        sample = sample[indices]
        distances = distances[indices]

        return sample, distances


# Experiment
class PointBasedExperimentResult(BaseModel):
    points: np.ndarray
    distances: np.ndarray
    
    @model_validator(mode='after')
    def validate_shapes(self):
        if not self.points.ndim == 2:
            raise ValueError(f"Points must be a 2D array, got {self.points.ndim}D")
        
        # Only accept 4D (x,y,z,shape) points
        if self.points.shape[1] != 4:
            raise ValueError(f"Points must have shape (n, 4), got {self.points.shape}")
        
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
                 sdfs: List[Callable[[np.ndarray], np.ndarray]], 
                 bound_begin: Union[np.ndarray, Tuple[float, float, float]], 
                 bound_end: Union[np.ndarray, Tuple[float, float, float]],
                 sdf_shape_values: List[float] = None) -> None:
        super().__init__()
        # Ensure sdfs is a list even if a single SDF is provided
        self.sdfs = sdfs if isinstance(sdfs, list) else [sdfs]
        self.sample_strategy = sample_strategy
        self.bound_begin = bound_begin if isinstance(bound_begin, np.ndarray) else np.array(bound_begin)
        self.bound_end = bound_end if isinstance(bound_end, np.ndarray) else np.array(bound_end)
        self.sdf_shape_values = sdf_shape_values
        
    def do(self) -> PointBasedExperimentResult:
        """Perform the experiment."""
        points, distances = self.sample_strategy.sample(self.sdfs, self.sdf_shape_values)
        return PointBasedExperimentResult(points=points, distances=distances)


if __name__ == "__main__":
    # Example usage with 4D points
    def example_sphere_sdf(points: np.ndarray) -> np.ndarray:
        # Sphere SDF with radius depending on the shape parameter
        # 3D points with radius 1.0
        xyz = points[:, :3]
        return np.linalg.norm(xyz, axis=1) - 1.0
    
    def example_box_sdf(points: np.ndarray) -> np.ndarray:
        # Box SDF with size 2x2x2
        xyz = points[:, :3]
        d = np.abs(xyz) - np.array([1.0, 1.0, 1.0])
        outside_distance = np.linalg.norm(np.maximum(d, 0), axis=1)
        inside_distance = np.minimum(np.max(d, axis=1), 0)
        return outside_distance + inside_distance
    
    # Define the bounds for 4D space
    bound_begin = np.array([-2.0, -2.0, -2.0, 0.0])  # x, y, z, shape
    bound_end = np.array([2.0, 2.0, 2.0, 1.0])       # x, y, z, shape
    
    # Define shape values
    sdf_shape_values = [0.0, 1.0]  # 0.0 = sphere, 1.0 = box
    
    # Create a 4D sample strategy
    sample_strategy = RandomSampleStrategy(
        n=1000, 
        bound_begin=bound_begin, 
        bound_end=bound_end,
        shape_values=sdf_shape_values
    )
    
    # Create the experiment with multiple SDFs
    experiment = PointBasedExperiment(
        sample_strategy=sample_strategy, 
        sdfs=[example_sphere_sdf, example_box_sdf], 
        bound_begin=bound_begin, 
        bound_end=bound_end,
        sdf_shape_values=sdf_shape_values
    )
    
    print("Dump Path:")
    print(experiment.dump_path)
    
    print()

    print("Doing Experiment:")
    result = experiment.do()
    print(result)
