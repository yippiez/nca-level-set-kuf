# %% [markdown]
# # SDF Visualization and CSG Rendering Demo (Quick Version)
# 
# This notebook demonstrates:
# - 3D signed distance functions (SDFs) with CSG-style OpenSCAD rendering
# - Static canonical 3D representations

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
    sdf_render_csg
)
from util.types import CSGRenderConfig

# Set up the output directory following notebook-specific convention
OUTPUT_DIR = get_reports_dir('sdf_demos_quick')

# %% [markdown]
# ## Basic SDFs with CSG Rendering

# %%
# Create and render a sphere using CSG style
def sphere_sdf(points):
    return sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.7)

config = CSGRenderConfig(
    grid_size=80,
    bounds=(-1.0, 1.0),
    save_path=str(OUTPUT_DIR / "sphere_csg.png"),
    image_size=(800, 800),
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield"
)

print("Rendering sphere...")
sphere_path = sdf_render_csg(sphere_sdf, config)
print(f"Sphere CSG image saved to: {sphere_path}")

# %%
# Create and render a box using CSG style
def box_sdf(points):
    return sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.8, 0.6, 0.4]))

config.save_path = str(OUTPUT_DIR / "box_csg.png")
print("Rendering box...")
box_path = sdf_render_csg(box_sdf, config)
print(f"Box CSG image saved to: {box_path}")

# %%
# Create and render a torus using CSG style
def torus_sdf(points):
    return sdf_torus(points, center=np.array([0.0, 0.0, 0.0]), r_major=0.5, r_minor=0.2)

config.save_path = str(OUTPUT_DIR / "torus_csg.png")
print("Rendering torus...")
torus_path = sdf_render_csg(torus_sdf, config)
print(f"Torus CSG image saved to: {torus_path}")

# %%
# Create and render a pill using CSG style
def pill_sdf(points):
    return sdf_pill(points, p1=np.array([0.0, 0.0, -0.7]), p2=np.array([0.0, 0.0, 0.7]), radius=0.3)

config.save_path = str(OUTPUT_DIR / "pill_csg.png")
print("Rendering pill...")
pill_path = sdf_render_csg(pill_sdf, config)
print(f"Pill CSG image saved to: {pill_path}")

# %% [markdown]
# ## Complex Compound Shapes

# %%
# Create a sphere with a box cut out
def sphere_with_box_cut(points):
    sphere = sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.8)
    box = sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.6, 0.6, 1.2]))
    # Subtraction: max(sphere, -box)
    return np.maximum(sphere, -box)

config.save_path = str(OUTPUT_DIR / "sphere_with_box_cut_csg.png")
config.camera_rotation = (30, 30, 0)
print("Rendering sphere with box cut...")
combined_path = sdf_render_csg(sphere_with_box_cut, config)
print(f"Combined CSG image saved to: {combined_path}")

# %%
# Create a complex shape using multiple operations
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

config.save_path = str(OUTPUT_DIR / "complex_csg.png")
config.camera_rotation = (30, 25, 0)
print("Rendering complex shape...")
complex_path = sdf_render_csg(complex_sdf, config)
print(f"Complex CSG image saved to: {complex_path}")

# %% [markdown]
# ## Summary

# %%
# Display paths to all the generated files
print("\nGenerated CSG images:")
files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
for file in files:
    print(f"- {os.path.join(OUTPUT_DIR, file)}")

print("\nDemo completed! All CSG visualizations saved to:", OUTPUT_DIR)