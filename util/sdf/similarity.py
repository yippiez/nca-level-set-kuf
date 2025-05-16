"""Similarity metrics for comparing SDFs."""

import numpy as np
from typing import Callable, Tuple, List
from scipy import stats

def compare_sdf(sdf1: Callable, sdf2: Callable, bounds: Tuple[float, float] = (-1.5, 1.5), 
               num_points: int = 10000, method: str = 'mse') -> float:
    """Compare two SDF functions and return a similarity metric.
    
    Args:
        sdf1: First SDF function taking points of shape (N, 3) and returning distances
        sdf2: Second SDF function taking points of shape (N, 3) and returning distances
        bounds: Tuple of (min, max) bounds for sampling points
        num_points: Number of random points to sample
        method: Similarity metric to use:
                - 'mse': Mean squared error (lower is better)
                - 'mae': Mean absolute error (lower is better)
                - 'correlation': Pearson correlation (higher is better)
                - 'iou': Intersection over Union of binary volumes (higher is better)
                - 'chamfer': Chamfer distance between zero level sets (lower is better)
    
    Returns:
        Similarity metric value
    """
    # Generate random points
    points = np.random.uniform(bounds[0], bounds[1], (num_points, 3))
    
    # Evaluate both SDFs
    values1 = sdf1(points)
    values2 = sdf2(points)
    
    # Reshape if needed
    if isinstance(values1, np.ndarray) and len(values1.shape) > 1:
        values1 = values1.flatten()
    if isinstance(values2, np.ndarray) and len(values2.shape) > 1:
        values2 = values2.flatten()
    
    # Calculate similarity based on method
    if method == 'mse':
        return np.mean((values1 - values2) ** 2)
    
    elif method == 'mae':
        return np.mean(np.abs(values1 - values2))
    
    elif method == 'correlation':
        return stats.pearsonr(values1, values2)[0]
    
    elif method == 'iou':
        # Create binary volumes (inside vs outside)
        binary1 = values1 <= 0
        binary2 = values2 <= 0
        
        # Calculate intersection and union
        intersection = np.sum(np.logical_and(binary1, binary2))
        union = np.sum(np.logical_or(binary1, binary2))
        
        # Return IoU
        return intersection / union if union > 0 else 0.0
    
    elif method == 'chamfer':
        # Find points close to the zero level set
        threshold = 0.05
        near_surface1 = np.abs(values1) < threshold
        near_surface2 = np.abs(values2) < threshold
        
        if np.sum(near_surface1) == 0 or np.sum(near_surface2) == 0:
            return float('inf')  # No surfaces found
        
        # Extract surface points
        surface_points1 = points[near_surface1]
        surface_points2 = points[near_surface2]
        
        # Compute chamfer distance (simplified)
        # For each point in set 1, find distance to closest point in set 2
        dists_1_to_2 = np.min(np.sum((surface_points1[:, np.newaxis, :] - 
                                     surface_points2[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        
        # For each point in set 2, find distance to closest point in set 1
        dists_2_to_1 = np.min(np.sum((surface_points2[:, np.newaxis, :] - 
                                     surface_points1[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        
        # Return average chamfer distance
        return (np.mean(dists_1_to_2) + np.mean(dists_2_to_1)) / 2
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def compare_sdf_batch(sdf1: Callable, sdf_list: List[Callable], 
                     shape_values: List[float], method: str = 'iou') -> List[float]:
    """Compare a single SDF function against a batch of ground truth functions.
    
    Args:
        sdf1: SDF function to compare, taking points of shape (N, 3) and returning distances
        sdf_list: List of ground truth SDF functions 
        shape_values: List of shape parameter values corresponding to each SDF
        method: Similarity metric to use
    
    Returns:
        List of similarity values for each shape parameter
    """
    results = []
    
    for i, shape_val in enumerate(shape_values):
        # Create a wrapper for sdf1 that uses the specific shape parameter
        def sdf1_wrapper(points):
            # Add shape parameter as 4th dimension
            points_4d = np.concatenate([points, np.full((points.shape[0], 1), shape_val)], axis=1)
            return sdf1(points_4d)
        
        # Compare with the corresponding ground truth
        similarity = compare_sdf(sdf1_wrapper, sdf_list[i], method=method)
        results.append(similarity)
    
    return results