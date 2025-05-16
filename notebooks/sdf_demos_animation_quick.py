# %% [markdown]
# # SDF CSG Animation Demo (Quick Version)
# 
# This notebook demonstrates rotating CSG-style animations with fewer frames for faster testing.

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
OUTPUT_DIR = get_reports_dir('sdf_demos_animation_quick')

# %% [markdown]
# ## Simple Shape Animations (Quick)

# %%
# Create a rotating sphere animation with fewer frames
def sphere_sdf(points):
    return sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.7)

config = CSGRenderConfig(
    grid_size=50,  # Lower resolution for faster rendering
    bounds=(-1.0, 1.0),
    save_path=str(OUTPUT_DIR / "sphere_rotating_quick.gif"),
    image_size=(400, 400),  # Smaller images for faster rendering
    n_frames=12,  # Fewer frames for faster generation
    fps=10,
    colorscheme="Cornfield"
)

print("Creating sphere animation (quick)...")
sphere_gif = sdf_render_csg_animation(sphere_sdf, config)
print(f"Sphere animation saved to: {sphere_gif}")

# %% [markdown]
# ## Complex Shape Animation (Quick)

# %%
# Create an animation of a sphere with box subtracted
def sphere_minus_box(points):
    sphere = sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.8)
    box = sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.6, 0.6, 2.0]))
    # Subtraction: max(sphere, -box)
    return np.maximum(sphere, -box)

config.save_path = str(OUTPUT_DIR / "sphere_minus_box_quick.gif")
print("Creating sphere minus box animation (quick)...")
sphere_box_gif = sdf_render_csg_animation(sphere_minus_box, config)
print(f"Sphere minus box animation saved to: {sphere_box_gif}")

# %% [markdown]
# ## Compound Shape Animation (Quick)

# %%
# Create a simple compound shape
def compound_shape(points):
    # Torus with spheres
    torus = sdf_torus(points, center=np.array([0.0, 0.0, 0.0]), r_major=0.5, r_minor=0.2)
    sphere1 = sdf_sphere(points, center=np.array([0.5, 0.0, 0.0]), radius=0.25)
    sphere2 = sdf_sphere(points, center=np.array([-0.5, 0.0, 0.0]), radius=0.25)
    
    # Combine: torus union with spheres
    combined = np.minimum(torus, np.minimum(sphere1, sphere2))
    return combined

config.save_path = str(OUTPUT_DIR / "compound_shape_quick.gif")
config.n_frames = 18  # A bit more frames for the compound shape
print("Creating compound shape animation (quick)...")
compound_gif = sdf_render_csg_animation(compound_shape, config)
print(f"Compound shape animation saved to: {compound_gif}")

# %% [markdown]
# ## Summary

# %%
# Display all generated animations
print("\nGenerated CSG animations (quick version):")
gif_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.gif')])
for file in gif_files:
    print(f"- {os.path.join(OUTPUT_DIR, file)}")

print(f"\nAll animations saved to: {OUTPUT_DIR}")
print("\nNote: These are quick versions with fewer frames for faster testing.")
print("For full quality animations, use sdf_demos_animation.py")