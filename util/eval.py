"""Evaluation utilities for neural network models."""

import torch
import torch.nn as nn
from typing import List
from .types import ModelSummary, LayerInfo, LayerDetails


def fcnn_n_perceptrons(model: nn.Module) -> int:
    """Count the total number of perceptrons in a FCNN model."""
    total_perceptrons = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_perceptrons += module.out_features
    
    return total_perceptrons


def fcnn_layer_details(model: nn.Module) -> List[LayerDetails]:
    """Get detailed information about each layer in a FCNN model."""
    layer_details = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_details.append(LayerDetails(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                n_neurons=module.out_features
            ))
    
    return layer_details


def model_summary(model: nn.Module) -> ModelSummary:
    """Generate a comprehensive summary of a neural network model."""
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_perceptrons = 0
    layers = []
    
    # Count perceptrons and gather layer information
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_info = LayerInfo(
                name=name,
                type='Linear',
                input_size=module.in_features,
                output_size=module.out_features,
                parameters=module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
                perceptrons=module.out_features
            )
            layers.append(layer_info)
            total_perceptrons += module.out_features
    
    return ModelSummary(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        non_trainable_parameters=non_trainable_parameters,
        total_perceptrons=total_perceptrons,
        layers=layers
    )