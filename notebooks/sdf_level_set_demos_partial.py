# %% [markdown]
# # Partial Level Set Demo - Just the parts that work quickly

# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from util.sdf import sdf_render_level_set, sdf_render_level_set_grid
from util.types import CSGRenderConfig
from util.paths import get_reports_dir

# Create output directory
OUTPUT_DIR = get_reports_dir("sdf_level_set_demos")

# %% [markdown]
# ## Define morphing SDF

# %%
def morphing_sdf(points):
    """4D SDF that morphs between sphere and box based on 4th dimension."""
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
# ## Quick Animation

# %%
# Configure animation with fewer frames
config = CSGRenderConfig(
    grid_size=50,  # Lower resolution
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "morphing_animation.gif"),
    image_size=(500, 500),
    camera_rotation=(45, 20, 0),
    n_frames=15,  # Fewer frames
    fps=8,
    colorscheme="Cornfield"
)

# Create morphing animation
shape_values = np.linspace(0, 4, config.n_frames)
gif_path = sdf_render_level_set(morphing_sdf, config, shape_values)
print(f"Saved morphing animation to: {gif_path}")

# %% [markdown]
# ## Smaller Grid

# %%
# Configure smaller grid
config_grid = CSGRenderConfig(
    grid_size=50,  # Lower resolution
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "morphing_grid.png"),
    image_size=(400, 400),
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield"
)

# Create grid with 9 shape values for 3x3 grid
shape_values_grid = np.linspace(0, 4, 9).tolist()
grid_path = sdf_render_level_set_grid(morphing_sdf, config_grid, shape_values_grid)
print(f"Saved morphing grid to: {grid_path}")