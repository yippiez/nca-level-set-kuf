# %% [markdown]
# # SDF Visualization and CSG Rendering Demo
# 
# This notebook demonstrates:
# - 3D signed distance functions (SDFs) from the `util.sdf` module with matplotlib visualizations
# - CSG-style rendering using OpenSCAD for canonical 3D representations
# - Both static images and animated GIFs

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
from pathlib import Path

# Add the parent directory to the path so we can import the util module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.paths import get_reports_dir
from util.sdf import (
    sdf_sphere,
    sdf_pill,
    sdf_box,
    sdf_torus,
    sdf_render,
    sdf_render_csg,
    sdf_render_csg_animation
)
from util.types import SDFRenderConfig, CSGRenderConfig

# Set up the output directory following notebook-specific convention
OUTPUT_DIR = get_reports_dir('sdf_demos')

# %% [markdown]
# ## Helper Functions

# %%
def visualize_and_save_sdf(sdf_func, name, description):
    """
    Visualize an SDF function and save it as a GIF using matplotlib.
    
    Args:
        sdf_func: The SDF function to visualize (must accept points of shape (..., 3))
        name: Name for the output file
        description: Text description of the shape
    """
    print(f"Generating {name} visualization...")
    
    # Generate the animation
    save_path = os.path.join(OUTPUT_DIR, f"{name}_matplotlib.gif")
    
    # Check if the animation already exists
    if os.path.exists(save_path):
        print(f"Animation for {name} already exists at {save_path}")
        return
    
    # Create config for the animation
    config = SDFRenderConfig(
        grid_size=50,
        bounds=(-1.5, 1.5),
        n_frames=36,
        save_path=save_path,
        dpi=120,
        fps=10
    )
    
    # Create the animation using the config
    anim = sdf_render(sdf_func, config)
    
    print(f"Saved {name} animation to {save_path}")

# %% [markdown]
# ## Basic SDFs: Sphere, Box, Pill, Torus

# %%
# Create and visualize a sphere SDF
sphere_func = partial(sdf_sphere, center=np.array([0.0, 0.0, 0.0]), radius=1.0)

visualize_and_save_sdf(
    sphere_func,
    "sphere",
    "Sphere SDF with radius 1.0 at the origin"
)

# %% [markdown]
# ### CSG-Style Sphere Rendering

# %%
# Now render the sphere using CSG style
config = CSGRenderConfig(
    grid_size=80,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "sphere_csg.png"),
    image_size=(800, 800),
    camera_rotation=(45, 20, 0),
    show_edges=True
)

try:
    sphere_path = sdf_render_csg(sphere_func, config)
    print(f"Sphere CSG image saved to: {sphere_path}")
except RuntimeError as e:
    print(f"CSG rendering failed: {e}")

# %%
# Create and visualize a box SDF
box_func = partial(sdf_box, center=np.array([0.0, 0.0, 0.0]), dims=np.array([1.0, 1.5, 0.8]))

visualize_and_save_sdf(
    box_func,
    "box",
    "Box SDF with dimensions (1.0, 1.5, 0.8) at the origin"
)

# CSG-style box rendering
config.save_path = str(OUTPUT_DIR / "box_csg.png")
try:
    box_path = sdf_render_csg(box_func, config)
    print(f"Box CSG image saved to: {box_path}")
except RuntimeError as e:
    print(f"CSG rendering failed: {e}")

# %%
# Create and visualize a pill SDF
pill_func = partial(
    sdf_pill, 
    p1=np.array([0.0, 0.0, -1.0]), 
    p2=np.array([0.0, 0.0, 1.0]), 
    radius=0.5
)

visualize_and_save_sdf(
    pill_func,
    "pill",
    "Pill SDF with radius 0.5 along the z-axis"
)

# CSG-style pill rendering
config.save_path = str(OUTPUT_DIR / "pill_csg.png")
try:
    pill_path = sdf_render_csg(pill_func, config)
    print(f"Pill CSG image saved to: {pill_path}")
except RuntimeError as e:
    print(f"CSG rendering failed: {e}")

# %%
# Create and visualize a torus SDF
torus_func = partial(
    sdf_torus, 
    center=np.array([0.0, 0.0, 0.0]), 
    r_major=0.8, 
    r_minor=0.2
)

visualize_and_save_sdf(
    torus_func,
    "torus",
    "Torus SDF with major radius 0.8 and minor radius 0.2 at the origin"
)

# CSG-style torus rendering
config.save_path = str(OUTPUT_DIR / "torus_csg.png")
try:
    torus_path = sdf_render_csg(torus_func, config)
    print(f"Torus CSG image saved to: {torus_path}")
except RuntimeError as e:
    print(f"CSG rendering failed: {e}")

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
# ## Complex Compound Shapes

# %%
# Create a sphere with a box cut out
sphere_with_box_cut = partial(
    sdf_subtraction,
    base_func=partial(sdf_sphere, center=np.array([0.0, 0.0, 0.0]), radius=1.0),
    subtract_func=partial(sdf_box, center=np.array([0.0, 0.0, 0.0]), dims=np.array([1.2, 0.6, 0.6]))
)

# Visualize and save the compound shape
visualize_and_save_sdf(
    sphere_with_box_cut,
    "sphere_with_box_cut",
    "Sphere with a box cut out"
)

# CSG-style rendering
config.save_path = str(OUTPUT_DIR / "sphere_with_box_cut_csg.png")
try:
    combined_path = sdf_render_csg(sphere_with_box_cut, config)
    print(f"Combined CSG image saved to: {combined_path}")
except RuntimeError as e:
    print(f"CSG rendering failed: {e}")

# %% [markdown]
# ## CSG Animated Rendering

# %%
# Create a rotating GIF of the compound shape using CSG rendering
gif_config = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "combined_rotating_csg.gif"),
    image_size=(600, 600),
    n_frames=36,
    fps=15,
    show_edges=True
)

try:
    gif_path = sdf_render_csg_animation(sphere_with_box_cut, gif_config)
    print(f"Animated CSG GIF saved to: {gif_path}")
except RuntimeError as e:
    print(f"CSG animation failed: {e}")

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

# CSG rendering
config.save_path = str(OUTPUT_DIR / "pill_union_csg.png")
config.camera_rotation = (30, 45, 0)
try:
    pill_union_path = sdf_render_csg(pill_union, config)
    print(f"Pill union CSG image saved to: {pill_union_path}")
except RuntimeError as e:
    print(f"CSG rendering failed: {e}")

# %% [markdown]
# ## Complex Shape Composition

# %%
# Create a more complex shape using multiple operations
def complex_sdf(points):
    # Main sphere
    sphere1 = sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.6)
    
    # Three smaller spheres to subtract
    sphere2 = sdf_sphere(points, center=np.array([0.4, 0.0, 0.0]), radius=0.3)
    sphere3 = sdf_sphere(points, center=np.array([-0.4, 0.0, 0.0]), radius=0.3)
    sphere4 = sdf_sphere(points, center=np.array([0.0, 0.4, 0.0]), radius=0.3)
    
    # Central torus
    torus = sdf_torus(points, center=np.array([0.0, 0.0, 0.0]), r_major=0.4, r_minor=0.15)
    
    # Combine: main sphere minus three spheres, union with torus
    subtracted = np.maximum(sphere1, -np.minimum(sphere2, np.minimum(sphere3, sphere4)))
    return np.minimum(subtracted, torus)

# Matplotlib visualization
visualize_and_save_sdf(
    complex_sdf,
    "complex",
    "Complex shape with multiple operations"
)

# CSG rendering
config.save_path = str(OUTPUT_DIR / "complex_csg.png")
config.camera_rotation = (30, 25, 0)
try:
    complex_path = sdf_render_csg(complex_sdf, config)
    print(f"Complex CSG image saved to: {complex_path}")
except RuntimeError as e:
    print(f"CSG rendering failed: {e}")

# %% [markdown]
# ## Summary

# %%
# Display paths to all the generated files
print("\nGenerated files:")
for ext in ['.gif', '.png']:
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(ext)])
    if files:
        print(f"\n{ext.upper()[1:]} files:")
        for file in files:
            print(f"- {os.path.join(OUTPUT_DIR, file)}")

print("\nDemo completed! All visualizations saved to:", OUTPUT_DIR)