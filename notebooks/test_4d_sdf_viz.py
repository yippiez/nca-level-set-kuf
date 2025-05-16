# %% [markdown]
# # Testing 4D SDF Model and Visualization
# 
# This notebook loads the trained FCNN model from cache and tests the 4D SDF visualization.

# %%
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from models.fcnn import FCNN
from util.cache import cache_get_torch, cache_get_json, cache_get_pickle
from util.sdf import sdf_render_level_set
import pathlib

# Get project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# %%
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## Load the Trained Model from Cache

# %%
def load_model_from_cache(cache_name='sdf_4d_model'):
    """Load the trained model from cache."""
    # Load model parameters
    model_params = cache_get_json(f'{cache_name}_params')
    print(f"Model parameters: {model_params}")
    
    # Create model instance
    model = FCNN(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        output_size=model_params['output_size'],
        num_layers=model_params['num_layers']
    ).to(device)
    
    # Load state dict - try torch format first, then pickle
    try:
        state_dict = cache_get_torch(cache_name)
    except:
        try:
            state_dict = cache_get_pickle(cache_name)
        except:
            raise ValueError(f"Could not load model from cache {cache_name}")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded successfully from cache: {cache_name}")
    return model

# Load the model
model = load_model_from_cache()

# %% [markdown]
# ## Test the Model with Different Shape Values

# %%
def test_model_predictions(model, test_points, shape_values):
    """Test model predictions for specific points and shape values."""
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for shape in shape_values:
            # Prepare inputs
            shape_indices = np.ones((len(test_points), 1)) * shape
            inputs = np.hstack([test_points, shape_indices])
            inputs_tensor = torch.FloatTensor(inputs).to(device)
            
            # Get predictions
            predictions = model(inputs_tensor).cpu().numpy()
            results[shape] = predictions
    
    return results

# Test some specific points
test_points = np.array([
    [0, 0, 0],      # Center
    [0.5, 0, 0],    # Right
    [0, 0.5, 0],    # Up
    [0, 0, 0.5],    # Forward
    [0.3, 0.3, 0],  # Diagonal in XY plane
])

shape_values = [0, 0.5, 1, 2, 3]  # Including intermediate values
results = test_model_predictions(model, test_points, shape_values)

# Display results
print("Model predictions for test points:")
print("Point coordinates:")
for i, point in enumerate(test_points):
    print(f"  Point {i}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")

print("\nDistances for different shape values:")
for shape in shape_values:
    print(f"\n  Shape s={shape}:")
    for i, distance in enumerate(results[shape]):
        print(f"    Point {i}: {distance[0]:.4f}")

# %% [markdown]
# ## Visualize the 3D Level Sets

# %%
# Create the 3D level set visualization with default 5x5 grid
fig = sdf_render_level_set(
    model,
    shape_values=None,  # Use default 5x5 grid
    grid_size=50,
    bounds=(-1.5, 1.5),
    figsize=(20, 16),
    save_path=str(PROJECT_ROOT / 'reports' / 'sdf' / 'sdf_level_set_default.png')
)
plt.show()

# %% [markdown]
# ## Visualize Specific Shape Values

# %%
# Visualize specific shape values including s=0 and s=0.5
specific_shapes = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
fig_specific = sdf_render_level_set(
    model,
    shape_values=specific_shapes,
    grid_size=50,
    bounds=(-1.5, 1.5),
    figsize=(15, 15),
    save_path=str(PROJECT_ROOT / 'reports' / 'sdf' / 'sdf_level_set_specific.png')
)
plt.show()

# %% [markdown]
# ## Detailed 3D Comparison for s=0 and s=0.5

# %%
# Focus on just s=0 and s=0.5 for detailed 3D comparison
comparison_shapes = [0, 0.5]
fig_comparison = sdf_render_level_set(
    model,
    shape_values=comparison_shapes,
    grid_size=80,  # Higher resolution for detail
    bounds=(-1.5, 1.5),
    figsize=(12, 6),
    save_path=str(PROJECT_ROOT / 'reports' / 'sdf' / 'sdf_level_set_comparison_s0_s0.5.png')
)
plt.show()


# %% [markdown]
# ## Summary
# 
# The 4D SDF model has been successfully loaded from cache and visualized using the new `sdf_render_level_set` function. 
# The visualizations show:
# 
# 1. A 5x5 grid of 3D shapes from s=0 (Pill) to s=4 (beyond Torus)
# 2. Specific shape values including fractional values like s=0.5
# 3. Detailed 3D comparison between s=0 and s=0.5
# 
# The model successfully learned to interpolate between different SDF shapes, creating smooth 3D surfaces 
# for arbitrary shape parameter values.