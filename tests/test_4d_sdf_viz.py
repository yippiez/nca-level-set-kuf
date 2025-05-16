"""Test 4D SDF visualization functionality."""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fcnn import FCNN
from util.cache import cache_get_torch, cache_get_json, cache_get_pickle
from util.sdf import sdf_render_level_set


@pytest.fixture
def device():
    """Get the compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cached_model(device):
    """Load the cached 4D SDF model."""
    cache_name = 'sdf_4d_model'
    
    # Load model parameters
    model_params = cache_get_json(f'{cache_name}_params')
    
    # Create model instance
    model = FCNN(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        output_size=model_params['output_size'],
        num_layers=model_params['num_layers']
    ).to(device)
    
    # Load state dict - try torch format first, then pickle
    try:
        state_dict = cache_get_torch(cache_name)
    except:
        try:
            state_dict = cache_get_pickle(cache_name)
        except:
            raise ValueError(f"Could not load model from cache {cache_name}")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def test_model_predictions(cached_model, device):
    """Test model predictions for specific points and shape values."""
    test_points = np.array([
        [0, 0, 0],      # Center
        [0.5, 0, 0],    # Right
        [0, 0.5, 0],    # Up
        [0, 0, 0.5],    # Forward
        [0.3, 0.3, 0],  # Diagonal in XY plane
    ])
    
    shape_values = [0, 0.5, 1, 2, 3]
    results = {}
    
    with torch.no_grad():
        for shape in shape_values:
            # Prepare inputs
            shape_indices = np.ones((len(test_points), 1)) * shape
            inputs = np.hstack([test_points, shape_indices])
            inputs_tensor = torch.FloatTensor(inputs).to(device)
            
            # Get predictions
            predictions = cached_model(inputs_tensor).cpu().numpy()
            results[shape] = predictions
    
    # Basic sanity checks
    assert len(results) == len(shape_values)
    for shape in shape_values:
        assert results[shape].shape == (5, 1)


def test_sdf_render_level_set_default(cached_model):
    """Test the 4D SDF visualization with default parameters."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        fig = sdf_render_level_set(
            cached_model,
            shape_values=None,  # Use default 5x5 grid
            grid_size=30,  # Reduced for faster testing
            bounds=(-1.5, 1.5),
            figsize=(10, 8),
            save_path=tmp_file.name
        )
        
        # Check that file was created
        assert os.path.exists(tmp_file.name)
        
        # Check file size to ensure something was written
        file_size = os.path.getsize(tmp_file.name)
        assert file_size > 1000  # Should be at least 1KB
        
        # Clean up
        os.unlink(tmp_file.name)


def test_sdf_render_level_set_specific_shapes(cached_model):
    """Test visualization with specific shape values."""
    specific_shapes = [0, 0.5, 1, 2, 3]
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        fig = sdf_render_level_set(
            cached_model,
            shape_values=specific_shapes,
            grid_size=30,
            bounds=(-1.5, 1.5),
            figsize=(12, 6),
            save_path=tmp_file.name
        )
        
        # Check that file was created
        assert os.path.exists(tmp_file.name)
        
        # Check file size
        file_size = os.path.getsize(tmp_file.name)
        assert file_size > 1000
        
        # Clean up
        os.unlink(tmp_file.name)


def test_sdf_render_level_set_comparison(cached_model):
    """Test detailed comparison visualization."""
    comparison_shapes = [0, 0.5]
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        fig = sdf_render_level_set(
            cached_model,
            shape_values=comparison_shapes,
            grid_size=40,  # Higher resolution for comparison
            bounds=(-1.5, 1.5),
            figsize=(8, 4),
            save_path=tmp_file.name
        )
        
        # Check that file was created
        assert os.path.exists(tmp_file.name)
        
        # Check file size
        file_size = os.path.getsize(tmp_file.name)
        assert file_size > 1000
        
        # Clean up
        os.unlink(tmp_file.name)


def test_model_interpolation(cached_model, device):
    """Test that model can interpolate between shapes."""
    # Test point at origin
    test_point = np.array([[0, 0, 0]])
    
    # Test interpolation from shape 0 to shape 1
    interpolation_values = np.linspace(0, 1, 11)
    distances = []
    
    with torch.no_grad():
        for s in interpolation_values:
            shape_indices = np.ones((1, 1)) * s
            inputs = np.hstack([test_point, shape_indices])
            inputs_tensor = torch.FloatTensor(inputs).to(device)
            
            prediction = cached_model(inputs_tensor).cpu().numpy()[0, 0]
            distances.append(prediction)
    
    # Check that we got results for all interpolation values
    assert len(distances) == len(interpolation_values)
    
    # The distances should change smoothly (no huge jumps)
    for i in range(1, len(distances)):
        diff = abs(distances[i] - distances[i-1])
        assert diff < 0.5  # Reasonable threshold for smooth change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])