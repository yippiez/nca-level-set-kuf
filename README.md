# nca-level-set-kuf
Creating Level-Setted Signed Distance Functions using NCA &amp;&amp; || similar models

## Project Structure

```
nca-level-set-kuf/
├── models/          # Neural network model definitions
├── notebooks/       # Interactive notebooks for experiments
├── tests/           # Test suite using pytest
├── util/            # Utility modules (cache, eval, sdf, etc.)
├── cache/           # Cached data (not tracked in git)
└── reports/         # Generated visualizations (not tracked in git)
```

## Testing

The project uses pytest for testing. Tests are located in the `tests/` directory.

### Running Tests

Install dev dependencies:
```bash
uv add --optional dev pytest pytest-cov
```

Run all tests:
```bash
uv run pytest
```

Run specific test file:
```bash
uv run pytest tests/test_fcnn_n_perceptron.py -v
```

Run with coverage:
```bash
uv run pytest --cov=util --cov=models
```

## Development

### Code Style Guidelines

- Utility modules: Single-line docstrings, no print statements
- Notebooks: Handle all printing and visualization
- Tests: Use temporary files instead of creating report files

### Build & Run Commands

- Run Notebook: `uv run notebooks/<NotebookName>`
- Run Tests: `uv run pytest`
