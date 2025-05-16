"""Tests for the cache utility module."""

import os
import numpy as np
import pytest
from pathlib import Path

from util.cache import cache_save, cache_get_numpy, cache_exists


def test_cache_numpy():
    """Test saving and loading NumPy arrays through the cache utility."""
    # Create a random NumPy array
    np.random.seed(42)  # For reproducibility
    original_array = np.random.rand(10, 10, 3)
    
    # Save the array to cache
    cache_name = "test_random_numpy"
    cache_save(original_array, cache_name)
    
    # Check if the cache exists
    assert cache_exists(cache_name), "Cache file should exist after saving"
    
    # Load the array from cache
    loaded_array = cache_get_numpy(cache_name)
    
    # Verify the arrays are identical
    assert loaded_array.shape == original_array.shape, "Shapes should match"
    assert np.allclose(loaded_array, original_array), "Arrays should be identical"
    
    # Clean up - remove the cache file
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_file = cache_dir / f"{cache_name}.npy"
    if cache_file.exists():
        cache_file.unlink()


def test_cache_exists():
    """Test the cache_exists function."""
    # Non-existent cache should return False
    assert not cache_exists("non_existent_cache"), "Non-existent cache should return False"
    
    # Create a cache and test existence
    test_array = np.array([1, 2, 3])
    cache_name = "test_exists"
    cache_save(test_array, cache_name)
    
    assert cache_exists(cache_name), "Existing cache should return True"
    
    # Clean up
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_file = cache_dir / f"{cache_name}.npy"
    if cache_file.exists():
        cache_file.unlink()