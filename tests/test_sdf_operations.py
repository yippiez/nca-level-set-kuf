"""Tests for SDF operations functions."""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.sdf.definitions import sdf_sphere
from util.sdf.operations import sdf_union
from util.sdf.similarity import sdf_get_sampled_boolean_similarity

def test_sdf_union_identical_spheres():
    """Test that union of two identical spheres is the same as the original sphere."""
    # Create a sphere SDF
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.5
    
    sphere_sdf = lambda points: sdf_sphere(points, center, radius)
    
    # Create a union of the same sphere with itself
    union_sdf = sdf_union([sphere_sdf, sphere_sdf])
    
    # Test grid parameters
    point_start = (-1.0, -1.0, -1.0)
    point_end = (1.0, 1.0, 1.0)
    step_size = 0.1
    
    # Compare the union with the original sphere
    similarity = sdf_get_sampled_boolean_similarity(
        sphere_sdf, union_sdf, point_start, point_end, step_size
    )
    
    # Union of identical spheres should match the original sphere exactly
    assert similarity == 1.0

def test_sdf_union_different_spheres():
    """Test that union of two different spheres creates a new shape."""
    # Create two spheres at different locations
    sphere1_center = np.array([-0.5, 0.0, 0.0])
    sphere2_center = np.array([0.5, 0.0, 0.0])
    radius = 0.4
    
    sphere1_sdf = lambda points: sdf_sphere(points, sphere1_center, radius)
    sphere2_sdf = lambda points: sdf_sphere(points, sphere2_center, radius)
    
    # Create a union of the two spheres
    union_sdf = sdf_union([sphere1_sdf, sphere2_sdf])
    
    # Test grid parameters
    point_start = (-1.0, -1.0, -1.0)
    point_end = (1.0, 1.0, 1.0)
    step_size = 0.1
    
    # Compare the union with each individual sphere
    similarity1 = sdf_get_sampled_boolean_similarity(
        sphere1_sdf, union_sdf, point_start, point_end, step_size
    )
    
    similarity2 = sdf_get_sampled_boolean_similarity(
        sphere2_sdf, union_sdf, point_start, point_end, step_size
    )
    
    # Union should be different from either sphere alone
    assert similarity1 < 1.0
    assert similarity2 < 1.0
    
    # Union should contain both spheres (have some similarity with both)
    assert similarity1 > 0.0
    assert similarity2 > 0.0

def test_sdf_union_disconnected_spheres():
    """Test that union of disconnected spheres preserves both shapes."""
    # Create two spheres at locations far enough to be disconnected
    sphere1_center = np.array([-1.0, 0.0, 0.0])
    sphere2_center = np.array([1.0, 0.0, 0.0])
    radius = 0.3
    
    sphere1_sdf = lambda points: sdf_sphere(points, sphere1_center, radius)
    sphere2_sdf = lambda points: sdf_sphere(points, sphere2_center, radius)
    
    # Create a union of the two spheres
    union_sdf = sdf_union([sphere1_sdf, sphere2_sdf])
    
    # Create points to test at specific locations
    test_points = np.array([
        sphere1_center,  # Inside first sphere
        sphere2_center,  # Inside second sphere
        np.array([0.0, 0.0, 0.0])  # Between spheres
    ])
    
    # Evaluate SDFs at test points
    sphere1_values = sphere1_sdf(test_points)
    sphere2_values = sphere2_sdf(test_points)
    union_values = union_sdf(test_points)
    
    # At center of sphere 1, union should match sphere 1
    assert union_values[0] == sphere1_values[0]
    
    # At center of sphere 2, union should match sphere 2
    assert union_values[1] == sphere2_values[1]
    
    # At middle point, union should take the minimum of both SDFs
    assert np.isclose(union_values[2], min(sphere1_values[2], sphere2_values[2]))
    
    # Union values should be negative (inside) at sphere centers
    assert union_values[0] < 0
    assert union_values[1] < 0

def test_sdf_union_multiple_spheres():
    """Test that union works with more than two shapes."""
    # Create three spheres
    centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0])
    ]
    radius = 0.4
    
    sphere_sdfs = [lambda points, c=c: sdf_sphere(points, c, radius) for c in centers]
    
    # Create a union of all three spheres
    union_sdf = sdf_union(sphere_sdfs)
    
    # Test at specific points inside each sphere
    test_points = np.array(centers)
    
    # Evaluate the union at test points
    union_values = union_sdf(test_points)
    
    # All points should be inside the union (negative distance)
    assert np.all(union_values < 0)
    
    # For each point, the union value should match the value of the corresponding sphere
    for i, sdf in enumerate(sphere_sdfs):
        single_value = sdf(test_points[i:i+1])[0]
        assert np.isclose(union_values[i], single_value)