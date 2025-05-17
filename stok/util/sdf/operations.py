"""Operations for SDF functions.

This module provides operations for manipulating and combining SDF functions.
"""

from typing import List, Callable
import numpy as np


def sdf_union(sdfs: List[Callable[[np.ndarray], np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
    """Create a union of multiple SDF functions.
    
    This function takes a list of partially applied SDF functions (functions that take
    only coordinates as input and return distances) and combines them into a single 
    SDF function using the union operation.
    
    Args:
        sdfs: List of SDF functions, each taking point coordinates (x, y, z) and
             returning distances
             
    Returns:
        A single SDF function that computes the union of all input SDFs
    """
    def union_sdf(points: np.ndarray) -> np.ndarray:
        # Initialize with a large positive value
        min_distances = np.full(len(points), np.inf)
        
        # Find the minimum distance at each point
        for sdf in sdfs:
            distances = sdf(points)
            min_distances = np.minimum(min_distances, distances)
            
        return min_distances
    
    return union_sdf