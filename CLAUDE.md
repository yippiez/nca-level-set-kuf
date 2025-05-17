# CLAUDE.md - Guidelines for Claude

## Build & Run Commands
- Use `uv run pytest -v` to test

## Code Style Guidelines
- Always save final results (models, visualizations, etc.) to the reports directory
- Don't print as much in the functions defined outside notebooks

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
- **Tests**:
  - `tests/test_render.py`: Tests for SDF rendering functions (migrated from visualization notebooks)

## Version Control Notes
- Memorize to commit pyproject.toml together with uv.lock when commiting
- Use git cli instead of git tool and do a single line commit that has no fluff

## Command Tips
- Can use `exa --tree` command to get an tree like view of the files

## Experiment Tree Versioning
- When adding a new experiment in the experiment-tree folder, the version name of the class should be its <parent version name> + <n_children> + 1 to distinguish 11 from 1 - 1
- We will separate the numbers with letters like:
  - Root example will be v1
  - Children will be v1a and v1b
  - Grandchildren of v1a will be v1a1, v1a2 and so on.
  - Grandchildren of v1a3 will be v1a3a, v1a3b and so on.
  - And so on, to create a clear hierarchical versioning system

## Folder Structure Notes
- Project root contains key directories:
  - `stok/`: Main source code directory
    - `tree/`: Versioned experiment tracking
    - `util/`: Utility modules and helper functions
      - `sdf/`: Signed Distance Function implementations
      - `paths.py`: Path management utilities
  - `tests/`: Test suites for different modules
  - `reports/`: Output directory for experiment results and visualizations