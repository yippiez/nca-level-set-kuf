"""Test 4D SDF visualization functionality."""

import pytest
import torch
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fcnn import FCNN


@pytest.fixture
def device():
    """Get the compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_model(device):
    """Create a test 4D SDF model for testing."""
    # Create a small test model
    input_size = 4   # x, y, z, shape
    hidden_size = 32  # Smaller for testing
    output_size = 1   # SDF value
    num_layers = 3    # Smaller for testing
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    model = FCNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers
    ).to(device)
    
    model.eval()
    return model


def test_model_predictions(test_model, device):
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
            predictions = test_model(inputs_tensor).cpu().numpy()
            results[shape] = predictions
    
    # Basic sanity checks
    assert len(results) == len(shape_values)
    for shape in shape_values:
        assert results[shape].shape == (5, 1)




def test_model_interpolation(test_model, device):
    """Test that model can interpolate between shapes."""
    # Test point at origin
    test_point = np.array([[0, 0, 0]])
    
    # Test interpolation from shape 0 to shape 1
    interpolation_values = np.linspace(0, 1, 5)  # Reduced count for speed
    distances = []
    
    with torch.no_grad():
        for s in interpolation_values:
            shape_indices = np.ones((1, 1)) * s
            inputs = np.hstack([test_point, shape_indices])
            inputs_tensor = torch.FloatTensor(inputs).to(device)
            
            prediction = test_model(inputs_tensor).cpu().numpy()[0, 0]
            distances.append(prediction)
    
    # Check that we got results for all interpolation values
    assert len(distances) == len(interpolation_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])