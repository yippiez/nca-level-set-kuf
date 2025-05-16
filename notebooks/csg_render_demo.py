# %% [markdown]
# # CSG-Style SDF Rendering Demo
# This notebook demonstrates the new CSG-style rendering pipeline for SDFs

# %%
import sys
import os
from pathlib import Path
from util.paths import get_reports_dir

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from util.sdf import sdf_sphere, sdf_box, sdf_torus, sdf_pill, sdf_render_csg, sdf_render_csg_animation
from util.types import CSGRenderConfig

# Create reports directory for CSG images following notebook-specific convention
csg_dir = get_reports_dir('csg_render_demo')

# %% [markdown]
# ## Basic Sphere CSG Rendering

# %%
# Define a sphere SDF
def sphere_sdf(points):
    return sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.5)

# Configure CSG rendering
config = CSGRenderConfig(
    grid_size=80,
    bounds=(-1.0, 1.0),
    save_path=str(csg_dir / "sphere_csg.png"),
    image_size=(800, 800),
    camera_rotation=(45, 20, 0),
    show_edges=True
)

# Render sphere
sphere_path = sdf_render_csg(sphere_sdf, config)
print(f"Sphere CSG image saved to: {sphere_path}")

# %% [markdown]
# ## Box CSG Rendering

# %%
# Define a box SDF
def box_sdf(points):
    return sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.8, 0.6, 0.4]))

config.save_path = str(csg_dir / "box_csg.png")
box_path = sdf_render_csg(box_sdf, config)
print(f"Box CSG image saved to: {box_path}")

# %% [markdown]
# ## Torus CSG Rendering

# %%
# Define a torus SDF
def torus_sdf(points):
    return sdf_torus(points, center=np.array([0.0, 0.0, 0.0]), r_major=0.5, r_minor=0.2)

config.save_path = str(csg_dir / "torus_csg.png")
torus_path = sdf_render_csg(torus_sdf, config)
print(f"Torus CSG image saved to: {torus_path}")

# %% [markdown]
# ## Combined Shapes CSG Rendering

# %%
# Define a combined shape (sphere with box subtracted)
def combined_sdf(points):
    sphere = sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.6)
    box = sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.4, 0.4, 1.2]))
    # Subtraction: max(sphere, -box)
    return np.maximum(sphere, -box)

config.save_path = str(csg_dir / "combined_csg.png")
combined_path = sdf_render_csg(combined_sdf, config)
print(f"Combined CSG image saved to: {combined_path}")

# %% [markdown]
# ## Animated CSG Rendering

# %%
# Create a rotating GIF of the combined shape
gif_config = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.0, 1.0),
    save_path=str(csg_dir / "combined_rotating.gif"),
    image_size=(600, 600),
    n_frames=36,
    fps=15,
    show_edges=True
)

gif_path = sdf_render_csg_animation(combined_sdf, gif_config)
print(f"Animated GIF saved to: {gif_path}")

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

config.save_path = str(csg_dir / "complex_csg.png")
config.camera_rotation = (30, 25, 0)
complex_path = sdf_render_csg(complex_sdf, config)
print(f"Complex CSG image saved to: {complex_path}")

# %%
print("CSG rendering demo completed!")
print(f"All images saved to: {csg_dir}")