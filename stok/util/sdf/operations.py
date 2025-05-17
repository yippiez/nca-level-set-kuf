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


def sdf_intersection(sdfs: List[Callable[[np.ndarray], np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
    """Create an intersection of multiple SDF functions.
    
    This function takes a list of partially applied SDF functions (functions that take
    only coordinates as input and return distances) and combines them into a single 
    SDF function using the intersection operation.
    
    Args:
        sdfs: List of SDF functions, each taking point coordinates (x, y, z) and
             returning distances
             
    Returns:
        A single SDF function that computes the intersection of all input SDFs
    """
    def intersection_sdf(points: np.ndarray) -> np.ndarray:
        # Initialize with a large negative value
        max_distances = np.full(len(points), -np.inf)
        
        # Find the maximum distance at each point
        for sdf in sdfs:
            distances = sdf(points)
            max_distances = np.maximum(max_distances, distances)
            
        return max_distances
    
    return intersection_sdf


def sdf_subtraction(sdf_a: Callable[[np.ndarray], np.ndarray], 
                   sdf_b: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Create a subtraction of one SDF function from another.
    
    This function takes two partially applied SDF functions and performs
    the subtraction operation (A - B). Points are inside the resulting shape
    if they are inside shape A but outside shape B.
    
    Args:
        sdf_a: The base SDF function (A), taking point coordinates (x, y, z) and
               returning distances
        sdf_b: The SDF function to subtract (B), taking point coordinates (x, y, z) and
               returning distances
             
    Returns:
        A single SDF function that computes the subtraction of B from A
    """
    def subtraction_sdf(points: np.ndarray) -> np.ndarray:
        # Compute distances for both SDFs
        dist_a = sdf_a(points)
        dist_b = sdf_b(points)
        
        # Subtraction is implemented as intersecting A with the complement of B
        # Complement of B is achieved by negating its distance
        return np.maximum(dist_a, -dist_b)
    
    return subtraction_sdf