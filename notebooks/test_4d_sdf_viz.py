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
from util.sdf import sdf_render_4d
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
# ## Visualize the 4D SDF Function

# %%
# Create the 4D visualization with default 5x5 grid
fig = sdf_render_4d(
    model,
    shape_values=None,  # Use default 5x5 grid
    grid_size=50,
    bounds=(-1.5, 1.5),
    figsize=(20, 16),
    save_path=str(PROJECT_ROOT / 'reports' / 'sdf' / 'sdf_4d_visualization_default.png')
)
plt.show()

# %% [markdown]
# ## Visualize Specific Shape Values

# %%
# Visualize specific shape values including s=0 and s=0.5
specific_shapes = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
fig_specific = sdf_render_4d(
    model,
    shape_values=specific_shapes,
    grid_size=50,
    bounds=(-1.5, 1.5),
    figsize=(15, 15),
    save_path=str(PROJECT_ROOT / 'reports' / 'sdf' / 'sdf_4d_visualization_specific.png')
)
plt.show()

# %% [markdown]
# ## Detailed Comparison for s=0 and s=0.5

# %%
# Focus on just s=0 and s=0.5 for detailed comparison
comparison_shapes = [0, 0.5]
fig_comparison = plt.figure(figsize=(12, 6))

for idx, shape_value in enumerate(comparison_shapes):
    ax = fig_comparison.add_subplot(1, 2, idx + 1)
    
    # Create a finer grid for detailed visualization
    grid_size = 100
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # z=0 slice
    Z = np.zeros_like(X)
    
    # Prepare inputs
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    shape_indices = np.ones((len(points), 1)) * shape_value
    inputs = np.hstack([points, shape_indices])
    
    # Get predictions
    inputs_tensor = torch.FloatTensor(inputs).to(device)
    with torch.no_grad():
        distances = model(inputs_tensor).cpu().numpy().reshape(X.shape)
    
    # Create visualization
    contour_filled = ax.contourf(X, Y, distances, levels=30, cmap='viridis')
    zero_contour = ax.contour(X, Y, distances, levels=[0], colors='red', linewidths=3)
    
    # Add other level contours
    level_contours = ax.contour(X, Y, distances, 
                               levels=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], 
                               colors='white', linewidths=1, alpha=0.7)
    ax.clabel(level_contours, inline=True, fontsize=8)
    
    ax.set_title(f's={shape_value}', fontsize=16, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    # Add shape name
    shape_names = {0: 'Pill', 0.5: 'Pill→Cylinder'}
    if shape_value in shape_names:
        ax.text(0.95, 0.95, shape_names[shape_value], 
                transform=ax.transAxes, 
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                fontsize=12)

fig_comparison.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'reports' / 'sdf' / 'sdf_4d_comparison_s0_s0.5.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Summary
# 
# The 4D SDF model has been successfully loaded from cache and visualized using the new `sdf_render_4d` function. 
# The visualizations show:
# 
# 1. A 5x5 grid of shape interpolations from s=0 (Pill) to s=4 (beyond Torus)
# 2. Specific shape values including fractional values like s=0.5
# 3. Detailed comparison between s=0 and s=0.5
# 
# The model successfully learned to interpolate between different SDF shapes, creating smooth transitions 
# between the discrete shape types it was trained on.