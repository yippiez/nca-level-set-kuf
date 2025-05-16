# %% [markdown]
# # Level Set Rendering Demonstrations
# This notebook demonstrates the level set rendering functions that morph shapes based on shape parameters.

# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from util.sdf import sdf_render_level_set, sdf_render_level_set_grid
from util.types import CSGRenderConfig
from util.paths import get_reports_dir, get_project_root
import os

# Create output directory
OUTPUT_DIR = get_reports_dir("sdf_level_set_demos")

# %% [markdown]
# ## Define a 4D SDF that morphs between sphere and box

# %%
def morphing_sdf(points):
    """4D SDF that morphs between sphere and box based on 4th dimension."""
    # Extract x, y, z, and shape parameter
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    shape = points[:, 3]
    
    # Sphere distance
    sphere_dist = np.sqrt(x**2 + y**2 + z**2) - 0.8
    
    # Box distance
    box_abs = np.maximum(np.maximum(np.abs(x) - 0.8, np.abs(y) - 0.8), np.abs(z) - 0.8)
    box_dist = np.maximum(box_abs, 0.0)
    
    # Morph between sphere and box
    return (1 - shape) * sphere_dist + shape * box_dist

# %% [markdown]
# ## Level Set Animation
# Create an animation that shows the shape morphing as the shape parameter changes

# %%
# Configure animation
config = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "morphing_animation.gif"),
    image_size=(600, 600),
    camera_rotation=(45, 20, 0),
    n_frames=30,
    fps=15,
    colorscheme="Cornfield"
)

# Create morphing animation
shape_values = np.linspace(0, 4, config.n_frames)
gif_path = sdf_render_level_set(morphing_sdf, config, shape_values)
print(f"Saved morphing animation to: {gif_path}")

# %% [markdown]
# ## Level Set Grid
# Create a grid showing different shape values

# %%
# Configure grid
config_grid = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "morphing_grid.png"),
    image_size=(400, 400),
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield"
)

# Create grid with different shape values
shape_values_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
grid_path = sdf_render_level_set_grid(morphing_sdf, config_grid, shape_values_grid)
print(f"Saved morphing grid to: {grid_path}")

# %% [markdown] 
# ## More Complex Example: Sphere to Torus Morph

# %%
def sphere_to_torus_sdf(points):
    """4D SDF that morphs from sphere to torus."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    shape = points[:, 3]
    
    # Sphere distance
    sphere_dist = np.sqrt(x**2 + y**2 + z**2) - 0.8
    
    # Torus distance
    q = np.sqrt(x**2 + y**2)
    torus_dist = np.sqrt((q - 0.6)**2 + z**2) - 0.3
    
    # Morph between them
    return (1 - shape) * sphere_dist + shape * torus_dist

# %%
# Create animation for sphere to torus
config_torus = CSGRenderConfig(
    grid_size=80,  # Higher resolution for torus
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "sphere_to_torus.gif"),
    image_size=(600, 600),
    camera_rotation=(30, 30, 0),
    n_frames=36,
    fps=15,
    colorscheme="Starnight"
)

gif_path_torus = sdf_render_level_set(sphere_to_torus_sdf, config_torus)
print(f"Saved sphere to torus animation to: {gif_path_torus}")

# %%
# Create grid for sphere to torus
config_torus_grid = CSGRenderConfig(
    grid_size=70,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "sphere_to_torus_grid.png"),
    image_size=(400, 400),
    camera_rotation=(30, 30, 0),
    colorscheme="Starnight"
)

grid_path_torus = sdf_render_level_set_grid(sphere_to_torus_sdf, config_torus_grid)
print(f"Saved sphere to torus grid to: {grid_path_torus}")

# %% [markdown]
# ## Custom Shape Parameters
# Create an animation with custom shape parameter progression

# %%
# Non-linear shape progression (ease-in-out)
def ease_in_out(t):
    return t * t * (3 - 2 * t)

custom_shape_values = ease_in_out(np.linspace(0, 1, 40)) * 4

config_custom = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "morphing_easeinout.gif"),
    image_size=(600, 600),
    camera_rotation=(45, 20, 0),
    n_frames=40,
    fps=20,
    colorscheme="Nature"
)

gif_path_custom = sdf_render_level_set(morphing_sdf, config_custom, custom_shape_values)
print(f"Saved custom animation to: {gif_path_custom}")

# %% [markdown]
# ## Summary
# We've created several level set visualizations:
# 1. Basic morphing animation between sphere and box
# 2. Grid showing discrete shape parameter values
# 3. Sphere to torus morphing
# 4. Custom non-linear shape progression
#
# All outputs are saved to the `reports/level_set_demos/` directory.