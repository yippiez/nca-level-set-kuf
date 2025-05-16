"""Tests for level set rendering functions."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from util.sdf import sdf_render_level_set, sdf_render_level_set_grid
from util.types import CSGRenderConfig


def simple_4d_sdf(points):
    """Simple 4D SDF for testing."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    shape = points[:, 3]
    
    # Sphere distance
    sphere_dist = np.sqrt(x**2 + y**2 + z**2) - 0.5
    
    # Box distance
    box_dist = np.maximum(np.maximum(np.abs(x) - 0.5, np.abs(y) - 0.5), np.abs(z) - 0.5)
    
    # Morph between them
    return (1 - shape) * sphere_dist + shape * box_dist


def test_level_set_animation():
    """Test level set animation creation."""
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
        config = CSGRenderConfig(
            grid_size=30,  # Low resolution for fast test
            bounds=(-1.0, 1.0),
            save_path=tmp.name,
            image_size=(200, 200),
            n_frames=5,
            fps=5
        )
        
        # Test with default shape values
        result_path = sdf_render_level_set(simple_4d_sdf, config)
        assert Path(result_path).exists()
        assert result_path.endswith('.gif')
        
        # Test with custom shape values
        shape_values = [0.0, 0.5, 1.0]
        result_path2 = sdf_render_level_set(simple_4d_sdf, config, shape_values)
        assert Path(result_path2).exists()


def test_level_set_grid():
    """Test level set grid creation."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        config = CSGRenderConfig(
            grid_size=30,
            bounds=(-1.0, 1.0),
            save_path=tmp.name,
            image_size=(150, 150)
        )
        
        # Test with default shape values
        result_path = sdf_render_level_set_grid(simple_4d_sdf, config)
        assert Path(result_path).exists()
        assert result_path.endswith('.png')
        
        # Test with custom shape values
        shape_values = [0.0, 0.33, 0.67, 1.0]
        result_path2 = sdf_render_level_set_grid(simple_4d_sdf, config, shape_values)
        assert Path(result_path2).exists()


def test_level_set_with_invalid_sdf():
    """Test error handling with invalid SDF function."""
    def invalid_sdf(points):
        # Wrong shape - doesn't expect 4D input
        return np.zeros(points.shape[0])
    
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
        config = CSGRenderConfig(
            grid_size=20,
            bounds=(-1.0, 1.0),
            save_path=tmp.name,
            image_size=(100, 100),
            n_frames=3
        )
        
        # Should raise an error
        with pytest.raises(Exception):
            sdf_render_level_set(invalid_sdf, config)


if __name__ == "__main__":
    test_level_set_animation()
    test_level_set_grid()
    print("All tests passed!")