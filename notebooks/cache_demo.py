# %% [markdown]
# # Cache Utility Demo
# 
# This notebook demonstrates how to use the cache utility to save and load NumPy arrays.

# %%
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.cache import cache_save, cache_get_numpy, cache_exists

# %% [markdown]
# ## Create a Random NumPy Array

# %%
# Set a random seed for reproducibility
np.random.seed(42)

# Create a random 10x10 array
random_array = np.random.rand(10, 10)
print(f"Generated random array shape: {random_array.shape}")
print(f"First few values:\n{random_array[:3, :3]}")

# %% [markdown]
# ## Save the Array to Cache

# %%
# Save the array to cache
cache_path = cache_save(random_array, "random_numpy")
print(f"Saved array to: {cache_path}")

# %% [markdown]
# ## Load the Array from Cache

# %%
# Load the array from cache
loaded_array = cache_get_numpy("random_numpy")
print(f"Loaded array shape: {loaded_array.shape}")
print(f"First few values:\n{loaded_array[:3, :3]}")

# Check that the loaded array is identical to the original
is_identical = np.array_equal(random_array, loaded_array)
print(f"Original and loaded arrays are identical: {is_identical}")

