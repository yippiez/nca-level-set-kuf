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
- In notebooks the cache utility could be used to store inter variables in notebooks to not compute them again when running notebooks again
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
  - CSG rendering in `util/sdf/render.py`: `sdf_render_csg` and `sdf_render_csg_animation`
  - Visualization: `sdf_render` function using marching cubes algorithm from scikit-image
  - Operations: Union, Intersection, and Subtraction for combining shapes
- SDF visualizations are found in `reports/<notebook_name>/` following the folder convention

## Allowed Commands 
```json
{
  "allowed_command_prefixes": ["uv", "pytest", "git"]
}
``` 

## Version Control Notes
- Memorize to commit pyproject.toml together with uv.lock when commiting
- Use git cli instead of git tool and do a single line commit that has no fluff