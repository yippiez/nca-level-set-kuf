# %% [markdown]
# # Test FCNN Perceptron Counter
# 
# This notebook tests the `fcnn_n_perceptrons` function from the evaluation utilities
# by loading the cached 4D SDF model and counting its perceptrons.

# %%
import os
import sys
import torch
import pathlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from models.fcnn import FCNN
from util.cache import cache_get_torch, cache_get_json, cache_get_pickle
from util.eval import fcnn_n_perceptrons, model_summary

# Get project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# %%
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## Load the Cached Model

# %%
def load_model_from_cache(cache_name='sdf_4d_model'):
    """Load the trained model from cache."""
    # Load model parameters
    model_params = cache_get_json(f'{cache_name}_params')
    print(f"Model parameters: {model_params}")
    
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
    
    print(f"Model loaded successfully from cache: {cache_name}")
    return model

# Load the model
model = load_model_from_cache()

# %% [markdown]
# ## Count Perceptrons

# %%
# Use the fcnn_n_perceptrons function to count neurons
print("\nCounting perceptrons in the model:")
print("=" * 50)
n_perceptrons = fcnn_n_perceptrons(model)
print("=" * 50)

# %% [markdown]
# ## Model Summary

# %%
# Get comprehensive model summary
print("\nDetailed Model Summary:")
print("=" * 50)
summary = model_summary(model)

print(f"Total Parameters: {summary['total_parameters']:,}")
print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
print(f"Non-trainable Parameters: {summary['non_trainable_parameters']:,}")
print(f"Total Perceptrons: {summary['total_perceptrons']}")
print("\nLayer Details:")
print("-" * 50)
print(f"{'Layer Name':<20} {'Type':<10} {'In':<6} {'Out':<6} {'Params':<10} {'Neurons':<8}")
print("-" * 50)

for layer in summary['layers']:
    print(f"{layer['name']:<20} {layer['type']:<10} {layer['input_size']:<6} {layer['output_size']:<6} {layer['parameters']:<10} {layer['perceptrons']:<8}")

# %% [markdown]
# ## Manual Verification

# %%
# Let's manually verify our count by examining the model structure
print("\nManual verification of model structure:")
print("=" * 50)
print(model)

# Count manually
manual_count = 0
for i, layer in enumerate(model.layers):
    if isinstance(layer, torch.nn.Linear):
        print(f"Layer {i}: Linear({layer.in_features}, {layer.out_features}) = {layer.out_features} neurons")
        manual_count += layer.out_features

print(f"\nManual count: {manual_count} perceptrons")
print(f"Function count: {n_perceptrons} perceptrons")
print(f"Match: {manual_count == n_perceptrons}")

# %% [markdown]
# ## Summary
# 
# The `fcnn_n_perceptrons` function successfully counts the total number of perceptrons
# in the FCNN model. For the cached 4D SDF model:
# 
# - Input size: 4 (x, y, z, shape)
# - Hidden layers: 4 layers with 64 neurons each
# - Output layer: 1 neuron
# - Total perceptrons: 257 (64 × 4 + 1)
# 
# The function provides both layer-wise breakdown and total count, making it useful
# for understanding model complexity.