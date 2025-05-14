# %% [markdown]
# # SDF Visualization Notebook
# 
# This notebook demonstrates the 3D signed distance functions (SDFs) from the `util.sdf` module
# and generates visualizations as GIF files.

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time

# Add the parent directory to the path so we can import the util module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.sdf import (
    sdf_sphere,
    sdf_pill,
    sdf_box,
    sdf_torus,
    sdf_render
)

# Set up the output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'sdf')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ## Helper Functions

# %%
def visualize_and_save_sdf(sdf_func, name, description):
    """
    Visualize an SDF function and save it as a GIF.
    
    Args:
        sdf_func: The SDF function to visualize (must accept points of shape (..., 3))
        name: Name for the output file
        description: Text description of the shape
    """
    print(f"Generating {name} visualization...")
    
    # Generate the animation
    save_path = os.path.join(OUTPUT_DIR, f"{name}.gif")
    
    # Cache the calculation to avoid recomputing
    cache_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'cache',
        f"{name}_anim.npy"
    )
    
    # Check if the animation already exists
    if os.path.exists(save_path):
        print(f"Animation for {name} already exists at {save_path}")
        return
    
    # Create the animation
    anim = sdf_render(
        sdf_func,
        grid_size=50,
        bounds=(-1.5, 1.5),
        n_frames=36,
        save_path=save_path,
        dpi=120,
        fps=10
    )
    
    print(f"Saved {name} animation to {save_path}")

# %% [markdown]
# ## Sphere SDF

# %%
# Create a sphere SDF with center at origin and radius 1.0
sphere_func = partial(sdf_sphere, center=np.array([0.0, 0.0, 0.0]), radius=1.0)

# Visualize and save the sphere SDF
visualize_and_save_sdf(
    sphere_func,
    "sphere",
    "Sphere SDF with radius 1.0 at the origin"
)

# %% [markdown]
# ## Box SDF

# %%
# Create a box SDF with center at origin and dimensions (1.0, 1.5, 0.8)
box_func = partial(sdf_box, center=np.array([0.0, 0.0, 0.0]), dimensions=np.array([1.0, 1.5, 0.8]))

# Visualize and save the box SDF
visualize_and_save_sdf(
    box_func,
    "box",
    "Box SDF with dimensions (1.0, 1.5, 0.8) at the origin"
)

# %% [markdown]
# ## Pill SDF

# %%
# Create a pill SDF with axis from (0,0,-1) to (0,0,1) and radius 0.5
pill_func = partial(
    sdf_pill, 
    p1=np.array([0.0, 0.0, -1.0]), 
    p2=np.array([0.0, 0.0, 1.0]), 
    radius=0.5
)

# Visualize and save the pill SDF
visualize_and_save_sdf(
    pill_func,
    "pill",
    "Pill SDF with radius 0.5 along the z-axis"
)

# %% [markdown]
# ## Torus SDF

# %%
# Create a torus SDF with center at origin, major radius 0.8 and minor radius 0.2
torus_func = partial(
    sdf_torus, 
    center=np.array([0.0, 0.0, 0.0]), 
    major_radius=0.8, 
    minor_radius=0.2
)

# Visualize and save the torus SDF
visualize_and_save_sdf(
    torus_func,
    "torus",
    "Torus SDF with major radius 0.8 and minor radius 0.2 at the origin"
)

# %% [markdown]
# ## Compound SDF Functions

# %%
def sdf_union(point, sdf_funcs):
    """
    Union of multiple SDF functions (take the minimum).
    
    Args:
        point: Array of shape (..., 3) with point coordinates
        sdf_funcs: List of SDF functions
        
    Returns:
        The minimum distance from the point to any of the shapes
    """
    distances = [func(point) for func in sdf_funcs]
    return np.minimum.reduce(distances)

def sdf_intersection(point, sdf_funcs):
    """
    Intersection of multiple SDF functions (take the maximum).
    
    Args:
        point: Array of shape (..., 3) with point coordinates
        sdf_funcs: List of SDF functions
        
    Returns:
        The maximum distance from the point to any of the shapes
    """
    distances = [func(point) for func in sdf_funcs]
    return np.maximum.reduce(distances)

def sdf_subtraction(point, base_func, subtract_func):
    """
    Subtraction of one SDF from another.
    
    Args:
        point: Array of shape (..., 3) with point coordinates
        base_func: The base SDF function
        subtract_func: The SDF function to subtract
        
    Returns:
        The distance for the subtracted shape
    """
    return np.maximum(base_func(point), -subtract_func(point))

# %% [markdown]
# ## Create and Visualize Compound Shapes

# %%
# Create a sphere with a box cut out
sphere_with_box_cut = partial(
    sdf_subtraction,
    base_func=partial(sdf_sphere, center=np.array([0.0, 0.0, 0.0]), radius=1.0),
    subtract_func=partial(sdf_box, center=np.array([0.0, 0.0, 0.0]), dimensions=np.array([1.2, 0.6, 0.6]))
)

# Visualize and save the compound shape
visualize_and_save_sdf(
    sphere_with_box_cut,
    "sphere_with_box_cut",
    "Sphere with a box cut out"
)

# %%
# Create intersecting pills to form a plus-like shape
pill_x = partial(
    sdf_pill,
    p1=np.array([-1.0, 0.0, 0.0]),
    p2=np.array([1.0, 0.0, 0.0]),
    radius=0.3
)

pill_y = partial(
    sdf_pill,
    p1=np.array([0.0, -1.0, 0.0]),
    p2=np.array([0.0, 1.0, 0.0]),
    radius=0.3
)

pill_z = partial(
    sdf_pill,
    p1=np.array([0.0, 0.0, -1.0]),
    p2=np.array([0.0, 0.0, 1.0]),
    radius=0.3
)

pill_union = partial(sdf_union, sdf_funcs=[pill_x, pill_y, pill_z])

# Visualize and save the compound shape
visualize_and_save_sdf(
    pill_union,
    "pill_union",
    "Union of three perpendicular pills"
)

# %%
# Display paths to all the generated GIFs
print("\nGenerated GIF files:")
for file in sorted(os.listdir(OUTPUT_DIR)):
    if file.endswith('.gif'):
        print(f"- {os.path.join(OUTPUT_DIR, file)}")