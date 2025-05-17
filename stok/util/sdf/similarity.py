"""Similarity metrics for comparing SDFs."""

import numpy as np
from typing import Callable, Tuple

def sdf_get_sampled_boolean_similarity(
    sdf1: Callable[[np.ndarray], np.ndarray],
    sdf2: Callable[[np.ndarray], np.ndarray],
    point_start: Tuple[float, float, float],
    point_end: Tuple[float, float, float],
    step_size: float
) -> float:
    """
    Compare two SDFs based on inside/outside classification at sampled points.
    
    Samples points in a 3D grid from point_start to point_end with step_size,
    then compares whether the two SDFs agree on inside/outside classification.
    
    Args:
        sdf1: First SDF function that takes a batch of 3D points and returns distances
        sdf2: Second SDF function that takes a batch of 3D points and returns distances
        point_start: Starting point of the grid (x_min, y_min, z_min)
        point_end: Ending point of the grid (x_max, y_max, z_max)
        step_size: Step size for grid sampling
        
    Returns:
        Percentage (0.0 to 1.0) of points where both SDFs agree on inside/outside classification
    """
    # Create the 3D grid
    x = np.arange(point_start[0], point_end[0] + step_size, step_size)
    y = np.arange(point_start[1], point_end[1] + step_size, step_size)
    z = np.arange(point_start[2], point_end[2] + step_size, step_size)
    
    # Calculate total number of points
    total_points = len(x) * len(y) * len(z)
    
    # Create mesh grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Stack into points array for vectorized evaluation
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Evaluate both SDFs
    sdf1_values = sdf1(points)
    sdf2_values = sdf2(points)
    
    # Determine inside/outside classification (negative = inside, positive = outside)
    # SDF convention: negative means inside, positive means outside
    sdf1_inside = sdf1_values <= 0
    sdf2_inside = sdf2_values <= 0
    
    # Count matching classifications
    matches = (sdf1_inside == sdf2_inside)
    match_count = np.sum(matches)
    
    # Return percentage of matching points
    return float(match_count) / total_points