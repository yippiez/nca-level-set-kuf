import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import functools
from skimage.measure import marching_cubes
from typing import Union, Optional, List, Tuple
import torch.nn as nn
from ..types import SDFRenderConfig, SDFLevelSetConfig

def sdf_sphere(point: np.ndarray, center: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a sphere."""
    point = np.asarray(point)
    center = np.asarray(center)
    dist = np.linalg.norm(point - center, axis=-1)
    return dist - radius

def sdf_pill(point: np.ndarray, p1: np.ndarray, p2: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a pill shape."""
    point = np.asarray(point)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    
    axis = p2 - p1
    axis_len = np.linalg.norm(axis)
    axis_norm = axis / axis_len if axis_len > 0 else np.array([0, 0, 1])
    
    if point.ndim > 1:
        v = point - p1
        proj = np.sum(v * axis_norm, axis=-1)
        proj = np.clip(proj, 0, axis_len)
        closest = p1 + np.outer(proj, axis_norm)
        dist = np.linalg.norm(point - closest, axis=-1)
        return dist - radius
    else:
        v = point - p1
        proj = np.dot(v, axis_norm)
        proj = np.clip(proj, 0, axis_len)
        closest = p1 + proj * axis_norm
        dist = np.linalg.norm(point - closest)
        return dist - radius

def sdf_box(point: np.ndarray, center: np.ndarray, dims: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to an axis-aligned box."""
    point = np.asarray(point)
    center = np.asarray(center)
    dims = np.asarray(dims)
    
    half = dims / 2
    local = np.abs(point - center)
    d = local - half
    
    inside = np.min(half - local, axis=-1)
    inside = np.where(np.all(d < 0, axis=-1), -inside, 0)
    outside = np.sqrt(np.sum(np.maximum(d, 0)**2, axis=-1))
    
    return outside + inside

def sdf_torus(point: np.ndarray, center: np.ndarray, r_major: float, r_minor: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a torus aligned with the xz-plane."""
    point = np.asarray(point)
    center = np.asarray(center)
    
    local = point - center
    xz_dist = np.sqrt(local[..., 0]**2 + local[..., 2]**2) - r_major
    dist = np.sqrt(xz_dist**2 + local[..., 1]**2) - r_minor
    
    return dist
