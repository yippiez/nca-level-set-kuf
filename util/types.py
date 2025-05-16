"""Type definitions for the util module using Pydantic."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
import numpy as np
import torch.nn as nn


class LayerInfo(BaseModel):
    """Information about a single layer in a neural network."""
    name: str
    type: str
    input_size: int
    output_size: int
    parameters: int
    perceptrons: int


class ModelSummary(BaseModel):
    """Comprehensive summary of a neural network model."""
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    total_perceptrons: int
    layers: List[LayerInfo]


class LayerDetails(BaseModel):
    """Detailed information about a layer for fcnn_layer_details function."""
    name: str
    in_features: int
    out_features: int
    n_neurons: int


class CacheMetadata(BaseModel):
    """Metadata for cached objects."""
    cache_name: str
    timestamp: Optional[str] = None
    size_bytes: Optional[int] = None
    format: str = "pickle"
    metadata: Optional[Dict[str, Any]] = None


class SDFRenderConfig(BaseModel):
    """Configuration for SDF rendering."""
    grid_size: int = 50
    bounds: Tuple[float, float] = (-1.0, 1.0)
    threshold: float = 0.0
    n_frames: int = 36
    save_path: Optional[str] = None
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 100
    fps: int = 15


class SDFLevelSetConfig(BaseModel):
    """Configuration for SDF level set rendering."""
    shape_values: Optional[List[float]] = None
    grid_size: int = 50
    bounds: Tuple[float, float] = (-1.0, 1.0)
    figsize: Tuple[int, int] = (20, 16)
    save_path: Optional[str] = None


class CSGRenderConfig(BaseModel):
    """Configuration for CSG-style rendering."""
    grid_size: int = 100
    bounds: Tuple[float, float] = (-1.0, 1.0)
    save_path: Optional[str] = None
    image_size: Tuple[int, int] = (800, 800)
    camera_rotation: Tuple[float, float, float] = (45, 20, 0)
    camera_distance: float = 10
    resolution: int = 100
    colorscheme: Optional[str] = "Cornfield"
    dpi: int = 150
    n_frames: int = 36
    fps: int = 15
    show_edges: bool = True