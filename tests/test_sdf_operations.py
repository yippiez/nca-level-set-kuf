"""Tests for SDF operations functions."""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stok.util.sdf.definitions import sdf_sphere
from stok.util.sdf.operations import sdf_union, sdf_intersection, sdf_subtraction
from stok.util.sdf.similarity import sdf_get_sampled_boolean_similarity

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


# Tests for Intersection

def test_sdf_intersection_identical_spheres():
    """Test that intersection of two identical spheres is the same as the original sphere."""
    # Create a sphere SDF
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.5
    
    sphere_sdf = lambda points: sdf_sphere(points, center, radius)
    
    # Create an intersection of the same sphere with itself
    intersection_sdf = sdf_intersection([sphere_sdf, sphere_sdf])
    
    # Test grid parameters
    point_start = (-1.0, -1.0, -1.0)
    point_end = (1.0, 1.0, 1.0)
    step_size = 0.1
    
    # Compare the intersection with the original sphere
    similarity = sdf_get_sampled_boolean_similarity(
        sphere_sdf, intersection_sdf, point_start, point_end, step_size
    )
    
    # Intersection of identical spheres should match the original sphere exactly
    assert similarity == 1.0


def test_sdf_intersection_overlapping_spheres():
    """Test that intersection of overlapping spheres creates a smaller shape."""
    # Create two spheres with overlapping regions
    sphere1_center = np.array([-0.3, 0.0, 0.0])
    sphere2_center = np.array([0.3, 0.0, 0.0])
    radius = 0.5
    
    sphere1_sdf = lambda points: sdf_sphere(points, sphere1_center, radius)
    sphere2_sdf = lambda points: sdf_sphere(points, sphere2_center, radius)
    
    # Create an intersection of the two spheres
    intersection_sdf = sdf_intersection([sphere1_sdf, sphere2_sdf])
    
    # Create test points
    test_points = np.array([
        np.array([0.0, 0.0, 0.0]),      # Center of overlap (inside both)
        sphere1_center,                  # Center of sphere 1 (outside intersection)
        sphere2_center,                  # Center of sphere 2 (outside intersection)
        np.array([-0.7, 0.0, 0.0]),     # Inside sphere 1 only
        np.array([0.7, 0.0, 0.0])       # Inside sphere 2 only
    ])
    
    # Evaluate SDFs at test points
    sphere1_values = sphere1_sdf(test_points)
    sphere2_values = sphere2_sdf(test_points)
    intersection_values = intersection_sdf(test_points)
    
    # Center of overlap should be inside intersection (negative distance)
    assert intersection_values[0] < 0
    
    # Center of overlap should have the maximum of both SDFs
    assert np.isclose(intersection_values[0], max(sphere1_values[0], sphere2_values[0]))
    
    # Centers of each sphere should be outside intersection (positive distance)
    assert intersection_values[1] > 0
    assert intersection_values[2] > 0
    
    # Points inside only one sphere should be outside intersection
    assert intersection_values[3] > 0
    assert intersection_values[4] > 0


def test_sdf_intersection_non_overlapping_spheres():
    """Test that intersection of non-overlapping spheres is empty."""
    # Create two spheres far enough apart to not overlap
    sphere1_center = np.array([-1.5, 0.0, 0.0])
    sphere2_center = np.array([1.5, 0.0, 0.0])
    radius = 0.5
    
    sphere1_sdf = lambda points: sdf_sphere(points, sphere1_center, radius)
    sphere2_sdf = lambda points: sdf_sphere(points, sphere2_center, radius)
    
    # Create an intersection of the two spheres
    intersection_sdf = sdf_intersection([sphere1_sdf, sphere2_sdf])
    
    # Test grid parameters
    point_start = (-2.0, -1.0, -1.0)
    point_end = (2.0, 1.0, 1.0)
    step_size = 0.1
    
    # Create a grid of test points
    x = np.arange(point_start[0], point_end[0], step_size)
    y = np.arange(point_start[1], point_end[1], step_size)
    z = np.arange(point_start[2], point_end[2], step_size)
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    
    # Evaluate intersection at grid points
    intersection_values = intersection_sdf(grid_points)
    
    # All points should be outside the intersection (no overlap)
    assert np.all(intersection_values > 0)


# Tests for Subtraction

def test_sdf_subtraction_nonoverlapping():
    """Test that subtraction where shapes don't overlap leaves the first shape unchanged."""
    # Create two non-overlapping spheres
    sphere1_center = np.array([-1.0, 0.0, 0.0])
    sphere2_center = np.array([1.0, 0.0, 0.0])
    radius = 0.5
    
    sphere1_sdf = lambda points: sdf_sphere(points, sphere1_center, radius)
    sphere2_sdf = lambda points: sdf_sphere(points, sphere2_center, radius)
    
    # Create subtraction A - B
    subtraction_sdf = sdf_subtraction(sphere1_sdf, sphere2_sdf)
    
    # Test grid parameters
    point_start = (-2.0, -1.0, -1.0)
    point_end = (2.0, 1.0, 1.0)
    step_size = 0.1
    
    # Compare subtraction with the first sphere (should be identical)
    similarity = sdf_get_sampled_boolean_similarity(
        sphere1_sdf, subtraction_sdf, point_start, point_end, step_size
    )
    
    # With no overlap, the subtraction should leave sphere1 unchanged
    assert similarity == 1.0


def test_sdf_subtraction_overlapping():
    """Test that subtraction correctly removes the overlapping part."""
    # Create two overlapping spheres
    sphere1_center = np.array([0.0, 0.0, 0.0])  # Base shape at origin
    sphere2_center = np.array([0.3, 0.0, 0.0])  # Subtracted shape offset a bit
    radius = 0.5
    
    sphere1_sdf = lambda points: sdf_sphere(points, sphere1_center, radius)
    sphere2_sdf = lambda points: sdf_sphere(points, sphere2_center, radius)
    
    # Create subtraction A - B
    subtraction_sdf = sdf_subtraction(sphere1_sdf, sphere2_sdf)
    
    # Create test points
    test_points = np.array([
        sphere1_center,                  # Center of sphere 1 (now inside sphere 2, should be outside result)
        sphere2_center,                  # Center of sphere 2 (should be outside result)
        np.array([-0.4, 0.0, 0.0]),      # Inside sphere 1 but not sphere 2 (should be inside result)
        np.array([0.7, 0.0, 0.0])        # Outside both spheres (should be outside result)
    ])
    
    # Evaluate SDFs at test points
    subtraction_values = subtraction_sdf(test_points)
    
    # Center of sphere 1 should now be outside the result (positive distance)
    # This is because it's inside sphere 2 which is being subtracted
    assert subtraction_values[0] > 0
    
    # Center of sphere 2 should be outside the result
    assert subtraction_values[1] > 0
    
    # Point inside sphere 1 but not sphere 2 should be inside result
    assert subtraction_values[2] < 0
    
    # Point outside both spheres should be outside result
    assert subtraction_values[3] > 0


def test_sdf_subtraction_contained():
    """Test that subtraction works when the second shape is fully contained in the first."""
    # Create a large sphere and a smaller one inside it
    large_center = np.array([0.0, 0.0, 0.0])
    small_center = np.array([0.0, 0.0, 0.0])  # Same center
    large_radius = 1.0
    small_radius = 0.5
    
    large_sdf = lambda points: sdf_sphere(points, large_center, large_radius)
    small_sdf = lambda points: sdf_sphere(points, small_center, small_radius)
    
    # Create subtraction (large - small) resulting in a hollow sphere
    hollow_sdf = sdf_subtraction(large_sdf, small_sdf)
    
    # Create test points at different distances from center
    r_values = np.array([0.0, 0.3, 0.6, 0.9, 1.2])
    test_points = np.array([np.array([r, 0.0, 0.0]) for r in r_values])
    
    # Evaluate hollow sphere at test points
    hollow_values = hollow_sdf(test_points)
    
    # Points inside small sphere should be outside result
    assert hollow_values[0] > 0  # Center
    assert hollow_values[1] > 0  # r = 0.3
    
    # Points between small and large sphere should be inside result
    assert hollow_values[2] < 0  # r = 0.6
    assert hollow_values[3] < 0  # r = 0.9
    
    # Points outside large sphere should be outside result
    assert hollow_values[4] > 0  # r = 1.2