# CLAUDE.md - Guidelines for Claude

## Build & Run Commands
- Run Notebook `uv run notebooks/<NotebookName>`
- Use `uv run pytest -v` to test

## Code Style Guidelines
- Notebooks are `.py` NOT '.ipynb' files that use interactive notebooks i.e. `# %%` to designate and sepearte cells.
- `# %% [markdown]` is used for markdown cells.
- The notebooks should be designed in such a way to be runnable as normal python files. Meaning
if a computationally heavy section is present cache it in the `cache` folder for future use.
- Make sure to not repeat variable names in notebooks to ensure no variable collision occurs.
- In notebooks the cache utility should ONLY be used for intermediate computations, NOT for final results:
  - Use cache for heavy intermediate computations that can be reused across runs
  - Always save final results (models, visualizations, etc.) to the reports directory
  - Don't use cache for storing trained models - these should go in reports
- Don't print as much in the functions defined outside notebooks

## Folder Structure
- *cache:* Where notebook caches are stored
- *models:* Where model definitions are stored
- *notebooks:* Where experiments are defined using notebooks
- *reports:* Where resulting tables, figures and models are stored
  - Files should follow the `reports/<notebook_name>/` convention
  - Each notebook should save its outputs to its own subdirectory
- *util:* Utility scripts like visualizations, preprocessing etc.

## Implementation Notes
- **Signed Distance Functions (SDFs)**: Implemented in `util/sdf/` module
  - Shape definitions in `util/sdf/definitions.py`: `sdf_sphere`, `sdf_box`, `sdf_pill`, `sdf_torus`
  - CSG rendering in `util/sdf/render.py`: 
    - `sdf_render_csg`: Render single CSG image
    - `sdf_render_csg_animation`: Render rotating animation
    - `sdf_render_level_set`: Render morphing animation based on shape parameter
    - `sdf_render_level_set_grid`: Render grid of images with different shape values
  - Level set functions expect 4D SDFs: (x, y, z, shape) => distance
  - Visualization: Using marching cubes algorithm and OpenSCAD for canonical rendering
  - Operations: Union, Intersection, and Subtraction for combining shapes
- SDF visualizations are found in `reports/<notebook_name>/` following the folder convention
- **Path Utilities**: `util/paths.py` provides consistent directory management functions
  - `get_project_root()`: Returns the project root directory
  - `get_reports_dir(notebook_name)`: Returns the reports directory for a specific notebook
  - `get_cache_dir()`: Returns the cache directory
- **Notebooks**:
  - `notebooks/sdf_demos.py`: Combined demonstration of SDF visualization and CSG rendering
  - `notebooks/sdf_demos_animation.py`: Demonstration of rotating CSG animations (full quality)
  - `notebooks/sdf_demos_animation_quick.py`: Quick test version of animations
  - `notebooks/sdf_level_set_demos.py`: Demonstration of level set morphing animations and grids
  - `notebooks/sdf_level_set_demos_quick.py`: Quick test version of level set rendering
  - `notebooks/fcnn_basic_v1a1.py`: FCNN training experiment for learning SDFs
- **Tests**:
  - `tests/test_cache.py`: Tests for the cache utility module (migrated from notebook demo)

## Allowed Commands 
```json
{
  "allowed_command_prefixes": ["uv", "pytest", "git"]
}
``` 

## Version Control Notes
- Memorize to commit pyproject.toml together with uv.lock when commiting
- Use git cli instead of git tool and do a single line commit that has no fluff

## Command Tips
- Can use `exa --tree` command to get an tree like view of the files