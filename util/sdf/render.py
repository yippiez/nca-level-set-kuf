"""CSG-style rendering using SolidPython and OpenSCAD."""

import numpy as np
import trimesh
import tempfile
import subprocess
from typing import Union, Optional, Callable
from skimage.measure import marching_cubes
from PIL import Image
import os
from ..types import CSGRenderConfig, SDFRenderConfig
from solid2 import *
from pathlib import Path


def sdf_to_mesh(sdf_func: Callable, config: Union[SDFRenderConfig, CSGRenderConfig]) -> trimesh.Trimesh:
    """Convert an SDF function to a mesh using marching cubes."""
    x = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    y = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    z = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    distances = sdf_func(points).reshape(X.shape)
    
    spacing = (
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1)
    )
    
    verts, faces, normals, _ = marching_cubes(
        distances, 
        level=0.0,
        spacing=spacing
    )
    
    verts = verts + np.array([config.bounds[0], config.bounds[0], config.bounds[0]])
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh


def sdf_render_csg(sdf_func: Callable, config: Optional[CSGRenderConfig] = None) -> str:
    """Render an SDF as a CSG-style canonical image using OpenSCAD."""
    if config is None:
        config = CSGRenderConfig()
    
    # Convert SDF to mesh
    mesh = sdf_to_mesh(sdf_func, config)
    
    # Save mesh to STL temporarily
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_stl:
        mesh.export(tmp_stl.name)
        stl_path = tmp_stl.name
    
    try:
        # Check if openscad is available
        result = subprocess.run(['which', 'openscad'], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("OpenSCAD not found. Please install OpenSCAD first: sudo apt-get install openscad")
        
        # Create OpenSCAD file that imports the STL
        scad_content = f"""
        // CSG-style rendering of SDF
        $fn = {config.resolution};
        
        // Import the mesh
        translate([0, 0, 0])
        rotate([0, 0, 0])
        import("{stl_path}");
        """
        
        with tempfile.NamedTemporaryFile(suffix='.scad', delete=False, mode='w') as tmp_scad:
            tmp_scad.write(scad_content)
            scad_path = tmp_scad.name
        
        # Create output path
        if config.save_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        else:
            output_path = config.save_path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Render with OpenSCAD
        cmd = [
            'openscad',
            '--render',
            '--viewall',
            '--autocenter',
            f'--imgsize={config.image_size[0]},{config.image_size[1]}',
            f'--camera=0,0,0,{config.camera_rotation[0]},{config.camera_rotation[1]},{config.camera_rotation[2]},{config.camera_distance}',
            '--projection=ortho',
            '-o', output_path,
            scad_path
        ]
        
        if config.colorscheme:
            cmd.extend(['--colorscheme', config.colorscheme])
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Rendered CSG image with OpenSCAD: {output_path}")
        
        # Clean up temporary files
        os.unlink(stl_path)
        os.unlink(scad_path)
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenSCAD error: {e.stderr.decode()}")
    except Exception as e:
        # Cleanup if exists
        if 'stl_path' in locals() and os.path.exists(stl_path):
            os.unlink(stl_path)
        if 'scad_path' in locals() and os.path.exists(scad_path):
            os.unlink(scad_path)
        raise e


def sdf_render_csg_animation(sdf_func: Callable, config: Optional[CSGRenderConfig] = None) -> str:
    """Render a rotating animation of the SDF."""
    if config is None:
        config = CSGRenderConfig()
    
    frames = []
    original_save_path = config.save_path
    
    for i in range(config.n_frames):
        angle = 360 * i / config.n_frames
        config.camera_rotation = (angle, 20, 0)
        
        # Create temporary frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            config.save_path = tmp.name
            frame_path = sdf_render_csg(sdf_func, config)
            frames.append(Image.open(frame_path))
            os.unlink(frame_path)
    
    # Create GIF
    if original_save_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
    else:
        output_path = original_save_path.replace('.png', '.gif')
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//config.fps,
        loop=0
    )
    
    config.save_path = original_save_path
    return output_path