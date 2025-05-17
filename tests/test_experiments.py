"""Tests for the experiment_tree module."""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tree.experiment import ExperimentBase
from tree.point_based import PointBasedExperiment, PointSampleStrategy, RandomSampleStrategy
from util.sdf import sdf_sphere


class MockSampleStrategy(PointSampleStrategy):
    """Mock sampling strategy for testing."""
    
    def __init__(self, n: int = 8, bound_begin: np.ndarray = None, bound_end: np.ndarray = None) -> None:
        """Initialize with defaults for testing."""
        self.n = n
        self.bound_begin = bound_begin if bound_begin is not None else np.array([-1.0, -1.0, -1.0])
        self.bound_end = bound_end if bound_end is not None else np.array([1.0, 1.0, 1.0])
    
    def sample(self, sdf: callable) -> np.ndarray:
        """Return a mock dataset."""
        # Create a simple 2x2x2 grid
        x = np.linspace(self.bound_begin[0], self.bound_end[0], 2)
        y = np.linspace(self.bound_begin[1], self.bound_end[1], 2)
        z = np.linspace(self.bound_begin[2], self.bound_end[2], 2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Evaluate the SDF
        distances = sdf(points)
        distances = distances.reshape(-1, 1)
        
        # Combine points and distances
        combined = np.hstack((points, distances))
        
        return combined


def test_experiment_base():
    """Test that ExperimentBase cannot be instantiated."""
    with pytest.raises(TypeError):
        _ = ExperimentBase()


def test_point_based_experiment():
    """Test PointBasedExperiment."""
    # Create a simple SDF function
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.5
    
    def sdf_func(points):
        """SDF function that works with 3D and 4D inputs."""
        # If 4D points are provided, use only the spatial coordinates
        if points.ndim == 2 and points.shape[1] == 4:
            return sdf_sphere(points[:, :3], center, radius)
        else:
            return sdf_sphere(points, center, radius)
    
    # Create the experiment
    bound_begin = np.array([-1.0, -1.0, -1.0])
    bound_end = np.array([1.0, 1.0, 1.0])
    
    experiment = PointBasedExperiment(
        sample_strategy=MockSampleStrategy(bound_begin=bound_begin, bound_end=bound_end),
        sdf=sdf_func,
        bound_begin=bound_begin,
        bound_end=bound_end
    )
    
    # Since the point_based.py has an unimplemented do() method,
    # we can't fully test it here. So we'll just confirm it can be instantiated.
    assert isinstance(experiment, PointBasedExperiment)
    assert experiment.sdf == sdf_func
    assert np.array_equal(experiment.bound_begin, bound_begin)
    assert np.array_equal(experiment.bound_end, bound_end)


def test_random_sample_strategy():
    """Test RandomSampleStrategy."""
    # Create a simple SDF function
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.5
    
    def sdf_func(points):
        """SDF function that works with 3D inputs."""
        return sdf_sphere(points, center, radius)
    
    # Create the strategy
    bound_begin = np.array([-1.0, -1.0, -1.0])
    bound_end = np.array([1.0, 1.0, 1.0])
    strategy = RandomSampleStrategy(n=100, bound_begin=bound_begin, bound_end=bound_end)
    
    # Sample points
    points, distances = strategy.sample(sdf_func)
    
    # Check results
    assert points.shape[0] == 100  # 100 points
    assert points.shape[1] == 3  # x, y, z
    assert distances.shape[0] == 100  # 100 distance values
    assert np.all(points >= bound_begin)  # All points within bounds
    assert np.all(points <= bound_end)
