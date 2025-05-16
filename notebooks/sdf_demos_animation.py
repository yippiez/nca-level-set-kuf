# %% [markdown]
# # SDF CSG Animation Demo
# 
# This notebook demonstrates rotating CSG-style animations of various SDF shapes.

# %%
import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the util module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.paths import get_reports_dir
from util.sdf import (
    sdf_sphere,
    sdf_pill,
    sdf_box,
    sdf_torus,
    sdf_render_csg_animation
)
from util.types import CSGRenderConfig

# Set up the output directory following notebook-specific convention
OUTPUT_DIR = get_reports_dir('sdf_demos_animation')

# %% [markdown]
# ## Simple Shape Animations

# %%
# Create a rotating sphere animation
def sphere_sdf(points):
    return sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.7)

config = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.0, 1.0),
    save_path=str(OUTPUT_DIR / "sphere_rotating.gif"),
    image_size=(600, 600),
    n_frames=36,
    fps=15,
    colorscheme="Cornfield"
)

print("Creating sphere animation...")
sphere_gif = sdf_render_csg_animation(sphere_sdf, config)
print(f"Sphere animation saved to: {sphere_gif}")

# %%
# Create a rotating torus animation
def torus_sdf(points):
    return sdf_torus(points, center=np.array([0.0, 0.0, 0.0]), r_major=0.5, r_minor=0.2)

config.save_path = str(OUTPUT_DIR / "torus_rotating.gif")
print("Creating torus animation...")
torus_gif = sdf_render_csg_animation(torus_sdf, config)
print(f"Torus animation saved to: {torus_gif}")

# %% [markdown]
# ## Complex Shape Animations

# %%
# Create an animation of intersecting shapes
def intersecting_spheres(points):
    sphere1 = sdf_sphere(points, center=np.array([0.3, 0.0, 0.0]), radius=0.5)
    sphere2 = sdf_sphere(points, center=np.array([-0.3, 0.0, 0.0]), radius=0.5)
    sphere3 = sdf_sphere(points, center=np.array([0.0, 0.3, 0.0]), radius=0.5)
    # Return the union (minimum) of all spheres
    return np.minimum(sphere1, np.minimum(sphere2, sphere3))

config.save_path = str(OUTPUT_DIR / "intersecting_spheres_rotating.gif")
print("Creating intersecting spheres animation...")
intersecting_gif = sdf_render_csg_animation(intersecting_spheres, config)
print(f"Intersecting spheres animation saved to: {intersecting_gif}")

# %%
# Create an animation of a sphere with box subtracted
def sphere_minus_box(points):
    sphere = sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.8)
    box = sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.6, 0.6, 1.2]))
    # Subtraction: max(sphere, -box)
    return np.maximum(sphere, -box)

config.save_path = str(OUTPUT_DIR / "sphere_minus_box_rotating.gif")
config.n_frames = 48  # More frames for smoother animation
print("Creating sphere minus box animation...")
sphere_box_gif = sdf_render_csg_animation(sphere_minus_box, config)
print(f"Sphere minus box animation saved to: {sphere_box_gif}")

# %% [markdown]
# ## Different Camera Settings

# %%
# Create a pill animation with different camera angle
def pill_sdf(points):
    return sdf_pill(points, p1=np.array([0.0, -0.6, 0.0]), p2=np.array([0.0, 0.6, 0.0]), radius=0.3)

# Custom camera configuration - tilted view
config_tilted = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.0, 1.0),
    save_path=str(OUTPUT_DIR / "pill_tilted_rotating.gif"),
    image_size=(600, 600),
    n_frames=36,
    fps=15,
    colorscheme="Cornfield"
)

# We'll use the standard animation function with default camera rotation
print("Creating pill animation with standard rotation...")
pill_gif = sdf_render_csg_animation(pill_sdf, config_tilted)
print(f"Pill animation saved to: {pill_gif}")

# %% [markdown]
# ## Complex Compound Shape Animation

# %%
# Create a complex shape with multiple operations
def complex_compound_shape(points):
    # Base torus
    torus = sdf_torus(points, center=np.array([0.0, 0.0, 0.0]), r_major=0.6, r_minor=0.2)
    
    # Spheres to add detail
    sphere1 = sdf_sphere(points, center=np.array([0.6, 0.0, 0.0]), radius=0.25)
    sphere2 = sdf_sphere(points, center=np.array([-0.6, 0.0, 0.0]), radius=0.25)
    sphere3 = sdf_sphere(points, center=np.array([0.0, 0.6, 0.0]), radius=0.25)
    sphere4 = sdf_sphere(points, center=np.array([0.0, -0.6, 0.0]), radius=0.25)
    
    # Box to cut through
    box = sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.3, 0.3, 1.5]))
    
    # Combine: torus with spheres, minus the box
    spheres_union = np.minimum(sphere1, np.minimum(sphere2, np.minimum(sphere3, sphere4)))
    combined = np.minimum(torus, spheres_union)
    result = np.maximum(combined, -box)
    
    return result

config_complex = CSGRenderConfig(
    grid_size=80,  # Higher resolution for complex shape
    bounds=(-1.0, 1.0),
    save_path=str(OUTPUT_DIR / "complex_compound_rotating.gif"),
    image_size=(800, 800),
    n_frames=60,  # More frames for complex shape
    fps=20,
    colorscheme="Cornfield"
)

print("Creating complex compound shape animation...")
complex_gif = sdf_render_csg_animation(complex_compound_shape, config_complex)
print(f"Complex compound animation saved to: {complex_gif}")

# %% [markdown]
# ## Summary

# %%
# Display all generated animations
print("\nGenerated CSG animations:")
gif_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.gif')])
for file in gif_files:
    print(f"- {os.path.join(OUTPUT_DIR, file)}")

print(f"\nAll animations saved to: {OUTPUT_DIR}")
print("\nNote: GIF files can be viewed in a web browser or image viewer that supports animations.")