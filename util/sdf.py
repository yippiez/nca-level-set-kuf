import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import functools
from skimage.measure import marching_cubes
from typing import Union, Optional, List, Tuple
import torch.nn as nn
from .types import SDFRenderConfig, SDFLevelSetConfig

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

def sdf_render(sdf_func: callable, config: Optional[SDFRenderConfig] = None) -> animation.FuncAnimation:
    """Render a 3D signed distance function as a rotating animation."""
    if config is None:
        config = SDFRenderConfig()
    
    x = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    y = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    z = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    distances = sdf_func(points).reshape(X.shape)
    
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    spacing = (
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1)
    )
    
    verts, faces, normals, _ = marching_cubes(
        distances, 
        level=config.threshold,
        spacing=spacing
    )
    
    verts = verts + np.array([config.bounds[0], config.bounds[0], config.bounds[0]])
    
    mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], triangles=faces, Z=verts[:, 2], 
                           cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(config.bounds)
    ax.set_ylim(config.bounds)
    ax.set_zlim(config.bounds)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    def update(frame, mesh, verts, faces, ax):
        ax.clear()
        ax.view_init(elev=30, azim=frame * (360 / config.n_frames))
        mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], triangles=faces, Z=verts[:, 2], 
                               cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(config.bounds)
        ax.set_ylim(config.bounds)
        ax.set_zlim(config.bounds)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return [mesh]
    
    anim = animation.FuncAnimation(
        fig, update, frames=config.n_frames, fargs=(mesh, verts, faces, ax),
        interval=1000/config.fps, blit=False
    )
    
    if config.save_path:
        anim.save(config.save_path, writer='pillow', fps=config.fps, dpi=config.dpi)
    
    plt.close(fig)
    
    return anim


def sdf_render_level_set(model: nn.Module, config: Optional[SDFLevelSetConfig] = None) -> plt.Figure:
    """Render 3D level sets of a 4D SDF function (x, y, z, shape) -> distance."""
    import torch
    
    if config is None:
        config = SDFLevelSetConfig()
    
    if config.shape_values is None:
        rows, cols = 5, 5
        shape_values = np.linspace(0, 4, rows * cols)
    else:
        shape_values = config.shape_values
        total_shapes = len(shape_values)
        rows = int(np.ceil(np.sqrt(total_shapes)))
        cols = int(np.ceil(total_shapes / rows))
    
    fig = plt.figure(figsize=config.figsize)
    fig.suptitle('3D Level Sets of 4D SDF: (x, y, z, shape) → distance', fontsize=16, fontweight='bold')
    
    x = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    y = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    z = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    device = next(model.parameters()).device
    
    spacing = (
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1)
    )
    
    for idx, shape_value in enumerate(shape_values):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        shape_indices = np.ones((len(points), 1)) * shape_value
        inputs = np.hstack([points, shape_indices])
        
        inputs_tensor = torch.FloatTensor(inputs).to(device)
        
        with torch.no_grad():
            distances = model(inputs_tensor).cpu().numpy().reshape(X.shape)
        
        try:
            verts, faces, normals, _ = marching_cubes(
                distances, 
                level=0.0,
                spacing=spacing
            )
            
            verts = verts + np.array([config.bounds[0], config.bounds[0], config.bounds[0]])
            
            ax.plot_trisurf(verts[:, 0], verts[:, 1], triangles=faces, Z=verts[:, 2], 
                           cmap='viridis', edgecolor='none', alpha=0.8)
        except ValueError:
            ax.text(0.5, 0.5, 0.5, 'No surface', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
        
        ax.set_xlim(config.bounds)
        ax.set_ylim(config.bounds)
        ax.set_zlim(config.bounds)
        ax.set_box_aspect([1, 1, 1])
        
        shape_label = f's={shape_value:.1f}' if shape_value % 1 != 0 else f's={int(shape_value)}'
        ax.set_title(shape_label, fontsize=12, fontweight='bold', pad=1)
        
        shape_names = {0: 'Pill', 1: 'Cylinder', 2: 'Box', 3: 'Torus'}
        if int(shape_value) == shape_value and shape_value in shape_names:
            ax.text2D(0.95, 0.95, shape_names[int(shape_value)], 
                     transform=ax.transAxes,
                     verticalalignment='top', 
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     fontsize=10)
        
        ax.view_init(elev=20, azim=45)
        
        if row == rows - 1:
            ax.set_xlabel('X', fontsize=10)
        else:
            ax.set_xticklabels([])
            
        if col == 0:
            ax.set_ylabel('Y', fontsize=10)
        else:
            ax.set_yticklabels([])
            
        ax.set_zlabel('Z', fontsize=10)
        
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
    
    plt.tight_layout()
    
    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches='tight')
    
    return fig