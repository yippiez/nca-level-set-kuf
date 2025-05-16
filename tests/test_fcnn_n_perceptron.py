"""Test FCNN perceptron counting functionality."""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fcnn import FCNN
from util.cache import cache_get_torch, cache_get_json, cache_get_pickle
from util.eval import fcnn_n_perceptrons, fcnn_layer_details, model_summary


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
    
    # Load state dict
    try:
        state_dict = cache_get_torch(cache_name)
    except:
        state_dict = cache_get_pickle(cache_name)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def test_fcnn_n_perceptrons(cached_model):
    """Test the fcnn_n_perceptrons function."""
    n_perceptrons = fcnn_n_perceptrons(cached_model)
    
    # Expected value based on model architecture: 
    # 4 hidden layers with 64 neurons each + 1 output neuron = 257
    assert n_perceptrons == 257, f"Expected 257 perceptrons, got {n_perceptrons}"


def test_fcnn_layer_details(cached_model):
    """Test the fcnn_layer_details function."""
    layer_details = fcnn_layer_details(cached_model)
    
    # Should have 5 layers
    assert len(layer_details) == 5, f"Expected 5 layers, got {len(layer_details)}"
    
    # Check first layer
    assert layer_details[0].in_features == 4
    assert layer_details[0].out_features == 64
    assert layer_details[0].n_neurons == 64
    
    # Check middle layers
    for i in range(1, 4):
        assert layer_details[i].in_features == 64
        assert layer_details[i].out_features == 64
        assert layer_details[i].n_neurons == 64
    
    # Check output layer
    assert layer_details[4].in_features == 64
    assert layer_details[4].out_features == 1
    assert layer_details[4].n_neurons == 1


def test_model_summary(cached_model):
    """Test the model_summary function."""
    summary = model_summary(cached_model)
    
    # Check total perceptrons
    assert summary.total_perceptrons == 257
    
    # Check parameter counts
    assert summary.total_parameters == 12865
    assert summary.trainable_parameters == 12865
    assert summary.non_trainable_parameters == 0
    
    # Check layers
    assert len(summary.layers) == 5
    
    # Verify layer info
    for i, layer in enumerate(summary.layers):
        assert layer.type == 'Linear'
        if i == 0:
            assert layer.input_size == 4
            assert layer.output_size == 64
        elif i < 4:
            assert layer.input_size == 64
            assert layer.output_size == 64
        else:
            assert layer.input_size == 64
            assert layer.output_size == 1


def test_manual_verification_matches(cached_model):
    """Test that manual count matches function count."""
    n_perceptrons = fcnn_n_perceptrons(cached_model)
    
    # Manual count
    manual_count = 0
    for layer in cached_model.layers:
        if isinstance(layer, torch.nn.Linear):
            manual_count += layer.out_features
    
    assert manual_count == n_perceptrons, f"Manual count {manual_count} doesn't match function count {n_perceptrons}"


def test_simple_1_input_1_output():
    """Test perceptron counting for simplest architecture: 1 input -> 1 output."""
    model = torch.nn.Linear(1, 1)
    assert fcnn_n_perceptrons(model) == 1


def test_1_input_5_hidden_1_output():
    """Test perceptron counting for: 1 input -> 5 hidden -> 1 output."""
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    assert fcnn_n_perceptrons(model) == 5 + 1  # 5 hidden + 1 output = 6


def test_multilayer_perceptron_variations():
    """Test various MLP architectures."""
    # 2 -> 3 -> 4 -> 1
    model1 = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1)
    )
    assert fcnn_n_perceptrons(model1) == 3 + 4 + 1  # 8
    
    # 10 -> 20 -> 10 -> 5 -> 1
    model2 = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    assert fcnn_n_perceptrons(model2) == 20 + 10 + 5 + 1  # 36
    
    # 3 -> 1 (single layer with multiple inputs)
    model3 = torch.nn.Linear(3, 1)
    assert fcnn_n_perceptrons(model3) == 1


def test_deeper_networks():
    """Test deeper network architectures."""
    # Very deep but narrow: 1 -> 2 -> 2 -> 2 -> 2 -> 1
    layers = [torch.nn.Linear(1, 2), torch.nn.ReLU()]
    for _ in range(3):
        layers.extend([torch.nn.Linear(2, 2), torch.nn.ReLU()])
    layers.append(torch.nn.Linear(2, 1))
    
    deep_model = torch.nn.Sequential(*layers)
    assert fcnn_n_perceptrons(deep_model) == 2 + 2 + 2 + 2 + 1  # 9
    
    # Wide network: 5 -> 100 -> 1
    wide_model = torch.nn.Sequential(
        torch.nn.Linear(5, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )
    assert fcnn_n_perceptrons(wide_model) == 100 + 1  # 101


def test_layer_details_on_simple_network():
    """Test layer details on a simple network."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1)
    )
    layer_details = fcnn_layer_details(model)
    
    # Should have 2 linear layers
    assert len(layer_details) == 2
    
    # Check first layer: 2 inputs -> 3 outputs
    assert layer_details[0].in_features == 2
    assert layer_details[0].out_features == 3
    assert layer_details[0].n_neurons == 3
    
    # Check last layer: 3 inputs -> 1 output
    assert layer_details[1].in_features == 3
    assert layer_details[1].out_features == 1
    assert layer_details[1].n_neurons == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])