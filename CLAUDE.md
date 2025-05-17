# CLAUDE.md - Guidelines for Claude

## Build & Run Commands
- Run Notebook `uv run notebooks/<NotebookName>`
- Use `uv run pytest -v` to test

## Code Style Guidelines
- Notebooks are `.py` NOT '.ipynb' files that use interactive notebooks i.e. `# %%` to designate and sepearte cells.
- `# %% [markdown]` is used for markdown cells.
- The notebooks should be designed in such a way to be runnable as normal python files.
- Make sure to not repeat variable names in notebooks to ensure no variable collision occurs.
- Always save final results (models, visualizations, etc.) to the reports directory
- Don't print as much in the functions defined outside notebooks

## Folder Structure
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
- **Notebooks**:
  - `notebooks/fcnn_basic_v1a1.py`: FCNN training experiment for learning SDFs
  - `notebooks/fcnn_range_models.py`: FCNN models for learning different shape parameter ranges
- **Tests**:
  - `tests/test_render.py`: Tests for SDF rendering functions (migrated from visualization notebooks)

## Version Control Notes
- Memorize to commit pyproject.toml together with uv.lock when commiting
- Use git cli instead of git tool and do a single line commit that has no fluff

## Command Tips
- Can use `exa --tree` command to get an tree like view of the files
