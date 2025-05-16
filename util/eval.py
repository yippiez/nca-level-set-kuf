"""Evaluation utilities for neural network models."""

import torch
import torch.nn as nn
from typing import Union


def fcnn_n_perceptrons(model: nn.Module) -> int:
    """
    Count the total number of perceptrons (neurons) in a Fully Connected Neural Network.
    
    This function counts all neurons in linear layers, including both hidden and output neurons.
    
    Args:
        model: A PyTorch neural network model containing Linear layers
        
    Returns:
        Total number of perceptrons in the model
        
    Example:
        >>> from models.fcnn import FCNN
        >>> model = FCNN(input_size=4, hidden_size=64, output_size=1, num_layers=5)
        >>> n_perceptrons = fcnn_n_perceptrons(model)
        >>> print(f"Model has {n_perceptrons} perceptrons")
    """
    total_perceptrons = 0
    
    print("Layer-wise perceptron count:")
    print("-" * 40)
    
    # Iterate through all modules in the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Each Linear layer has 'out_features' number of perceptrons
            n_neurons = module.out_features
            total_perceptrons += n_neurons
            
            # Print layer information
            print(f"{name:<20} {module.in_features:>4} -> {module.out_features:>4} : {n_neurons:>4} neurons")
    
    print("-" * 40)
    print(f"Total perceptrons: {total_perceptrons}")
    
    return total_perceptrons


def model_summary(model: nn.Module) -> dict:
    """
    Generate a comprehensive summary of a neural network model.
    
    Args:
        model: A PyTorch neural network model
        
    Returns:
        Dictionary containing model statistics
    """
    summary = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'non_trainable_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'total_perceptrons': 0,
        'layers': []
    }
    
    # Count perceptrons and gather layer information
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_info = {
                'name': name,
                'type': 'Linear',
                'input_size': module.in_features,
                'output_size': module.out_features,
                'parameters': module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
                'perceptrons': module.out_features
            }
            summary['layers'].append(layer_info)
            summary['total_perceptrons'] += module.out_features
    
    return summary