"""Tests for SDF similarity functions."""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.sdf import (
    sdf_sphere,
    sdf_box,
    sdf_get_sampled_boolean_similarity
)


def test_boolean_similarity_identical_sdfs():
    """Test that identical SDFs have perfect boolean similarity."""
    radius = 0.5
    center = np.array([0.0, 0.0, 0.0])
    
    # Use the same SDF function for both
    sdf1 = lambda points: sdf_sphere(points, center, radius)
    sdf2 = lambda points: sdf_sphere(points, center, radius)
    
    # Test with a small grid
    point_start = (-1.0, -1.0, -1.0)
    point_end = (1.0, 1.0, 1.0)
    step_size = 0.2
    
    similarity = sdf_get_sampled_boolean_similarity(
        sdf1, sdf2, point_start, point_end, step_size
    )
    
    # For identical SDFs, all points should match (100% similarity)
    assert similarity == 1.0


def test_boolean_similarity_different_sdfs():
    """Test that different SDFs have partial boolean similarity."""
    # Create a sphere and a box
    sphere_center = np.array([0.0, 0.0, 0.0])
    box_center = np.array([0.0, 0.0, 0.0])
    box_dims = np.array([0.8, 0.8, 0.8])
    
    sphere_sdf = lambda points: sdf_sphere(points, sphere_center, 0.5)
    box_sdf = lambda points: sdf_box(points, box_center, box_dims)
    
    # Test with a small grid
    point_start = (-1.0, -1.0, -1.0)
    point_end = (1.0, 1.0, 1.0)
    step_size = 0.2
    
    similarity = sdf_get_sampled_boolean_similarity(
        sphere_sdf, box_sdf, point_start, point_end, step_size
    )
    
    # For different SDFs, we expect partial similarity (between 0 and 1)
    assert similarity > 0.0
    assert similarity < 1.0


def test_boolean_similarity_disjoint_sdfs():
    """Test that completely disjoint SDFs have lower boolean similarity."""
    # Create two spheres at different locations
    sphere1_center = np.array([-0.5, 0.0, 0.0])
    sphere2_center = np.array([0.5, 0.0, 0.0])
    
    sphere1_sdf = lambda points: sdf_sphere(points, sphere1_center, 0.3)
    sphere2_sdf = lambda points: sdf_sphere(points, sphere2_center, 0.3)
    
    # Test with a small grid
    point_start = (-1.0, -1.0, -1.0)
    point_end = (1.0, 1.0, 1.0)
    step_size = 0.2
    
    similarity = sdf_get_sampled_boolean_similarity(
        sphere1_sdf, sphere2_sdf, point_start, point_end, step_size
    )
    
    # For disjoint shapes, we expect partial similarity
    # (matches will be in empty regions away from both shapes)
    assert similarity > 0.0
    assert similarity < 1.0


def test_boolean_similarity_comparison():
    """Test that the similarity of a sphere with itself is higher than between different shapes."""
    # Create a test sphere
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.5
    
    # Same sphere (perfect match)
    sphere_sdf = lambda points: sdf_sphere(points, center, radius)
    
    # Test with the same grid as other tests
    point_start = (-1.0, -1.0, -1.0)
    point_end = (1.0, 1.0, 1.0)
    step_size = 0.2
    
    # Get similarity between identical spheres (should be 1.0)
    identical_similarity = sdf_get_sampled_boolean_similarity(
        sphere_sdf, sphere_sdf, point_start, point_end, step_size
    )
    
    # Create a box with similar size
    box_center = np.array([0.0, 0.0, 0.0])
    box_dims = np.array([0.8, 0.8, 0.8])
    box_sdf = lambda points: sdf_box(points, box_center, box_dims)
    
    # Get similarity between sphere and box
    sphere_box_similarity = sdf_get_sampled_boolean_similarity(
        sphere_sdf, box_sdf, point_start, point_end, step_size
    )
    
    # Identical shapes should have higher similarity than different shapes
    assert identical_similarity > sphere_box_similarity
    
    # Also compare two different spheres with different sizes
    small_sphere_sdf = lambda points: sdf_sphere(points, center, 0.3)
    large_sphere_sdf = lambda points: sdf_sphere(points, center, 0.7)
    
    # Get similarity between different sized spheres
    different_size_similarity = sdf_get_sampled_boolean_similarity(
        small_sphere_sdf, large_sphere_sdf, point_start, point_end, step_size
    )
    
    # Different sized spheres should have lower similarity than identical spheres
    assert identical_similarity > different_size_similarity
    
    # Print actual values to help understand the relationships
    print(f"\nSimilarity values:")
    print(f"Identical spheres: {identical_similarity:.4f}")
    print(f"Sphere vs Box: {sphere_box_similarity:.4f}")
    print(f"Small vs Large sphere: {different_size_similarity:.4f}")
