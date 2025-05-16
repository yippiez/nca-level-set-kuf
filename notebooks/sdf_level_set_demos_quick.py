# %% [markdown]
# # Level Set Rendering Quick Test
# A quick test of the level set rendering functions with reduced resolution for faster testing.

# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from util.sdf import sdf_render_level_set, sdf_render_level_set_grid
from util.types import CSGRenderConfig
from util.paths import get_reports_dir

# Create output directory
OUTPUT_DIR = get_reports_dir("sdf_level_set_demos_quick")

# %% [markdown]
# ## Simple Morphing Test

# %%
def simple_morph_sdf(points):
    """Simple 4D SDF that morphs between sphere and box."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    shape = points[:, 3]
    
    # Sphere distance
    sphere_dist = np.sqrt(x**2 + y**2 + z**2) - 0.8
    
    # Box distance (simple version)
    box_dist = np.maximum(np.maximum(np.abs(x) - 0.8, np.abs(y) - 0.8), np.abs(z) - 0.8)
    
    # Linear interpolation
    return (1 - shape) * sphere_dist + shape * box_dist

# %% [markdown]
# ## Quick Animation Test

# %%
# Quick animation with fewer frames
config = CSGRenderConfig(
    grid_size=40,  # Lower resolution
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "quick_morph.gif"),
    image_size=(400, 400),  # Smaller images
    camera_rotation=(45, 20, 0),
    n_frames=10,  # Fewer frames
    fps=5,  # Slower animation
    colorscheme="Cornfield"
)

# Test animation
gif_path = sdf_render_level_set(simple_morph_sdf, config)
print(f"Saved quick animation to: {gif_path}")

# %% [markdown]
# ## Quick Grid Test

# %%
# Quick grid with fewer samples
config_grid = CSGRenderConfig(
    grid_size=40,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "quick_grid.png"),
    image_size=(300, 300),  # Smaller tiles
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield"
)

# Test grid with just 3 values
shape_values_grid = [0.0, 2.0, 4.0]
grid_path = sdf_render_level_set_grid(simple_morph_sdf, config_grid, shape_values_grid)
print(f"Saved quick grid to: {grid_path}")

# %% [markdown]
# ## Done!
# Quick test completed. Check the `reports/level_set_demos_quick/` directory for results.