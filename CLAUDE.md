# CLAUDE.md - Guidelines for Claude

## Build & Run Commands
- Run Notebook `uv run notebooks/<NotebookName>`

## Code Style Guidelines
- Notebooks are `.py` NOT '.ipynb' files that use interactive notebooks i.e. `# %%` to designate and sepearte cells.
- `# %% [markdown]` is used for markdown cells.
- The notebooks should be designed in such a way to be runnable as normal python files. Meaning
if a computationally heavy section is present cache it in the `cache` folder for future use.
- Make sure to not repeat variable names in notebooks to ensure no variable collision occurs.

## Folder Structure
- *cache:* Where notebook caches are stored
- *models:* Where model definitions are stored
- *notebooks:* Where experiments are defined using notebooks
- *reports:* Where resulting tables, figures and models are stored
- *util:* Utility scripts like visualizations, preprocessing etc.

## Implementation Notes
- **Signed Distance Functions (SDFs)**: Implemented in `util/sdf.py` with proper vectorization
  - Basic shapes: `sdf_sphere`, `sdf_box`, `sdf_cylinder`, `sdf_torus`
  - Visualization: `sdf_render` function using marching cubes algorithm from scikit-image
  - Operations: Union, Intersection, and Subtraction for combining shapes
- SDF visualizations are found in `reports/sdf/` as GIF animations

## Allowed Commands 
```json
{
  "allowed_command_prefixes": ["uv", "pytest", "git"]
}
``` 
