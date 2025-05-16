"""CSG-style rendering using OpenSCAD."""

import numpy as np
import tempfile
import subprocess
from typing import Optional, Callable
from skimage.measure import marching_cubes
from PIL import Image
import os
from ..types import CSGRenderConfig
from pathlib import Path


def sdf_to_stl(sdf_func: Callable, config: CSGRenderConfig, filepath: str) -> None:
    """Convert an SDF function directly to STL format."""
    # Create a 3D grid to sample the SDF
    x = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    y = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    z = np.linspace(config.bounds[0], config.bounds[1], config.grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Sample the SDF
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    distances = sdf_func(points).reshape(X.shape)
    
    # Calculate spacing for marching cubes
    spacing = (
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1),
        (config.bounds[1] - config.bounds[0]) / (config.grid_size - 1)
    )
    
    # Use marching cubes to extract the isosurface
    verts, faces, normals, _ = marching_cubes(
        distances, 
        level=0.0,
        spacing=spacing
    )
    
    # Adjust vertices to the correct position in space
    verts = verts + np.array([config.bounds[0], config.bounds[0], config.bounds[0]])
    
    # Write STL file directly
    with open(filepath, 'w') as f:
        f.write("solid sdf_mesh\n")
        
        for face in faces:
            # Get vertices of the triangle
            v0, v1, v2 = verts[face]
            
            # Calculate normal (cross product)
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid sdf_mesh\n")


def sdf_render_csg(sdf_func: Callable, config: Optional[CSGRenderConfig] = None) -> str:
    """Render an SDF as a CSG-style canonical image using OpenSCAD."""
    if config is None:
        config = CSGRenderConfig()
    
    # Convert SDF to STL
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False, mode='w') as tmp_stl:
        stl_path = tmp_stl.name
    
    sdf_to_stl(sdf_func, config, stl_path)
    
    try:
        # Check if openscad is available
        result = subprocess.run(['which', 'openscad'], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("OpenSCAD not found. Please install OpenSCAD first: sudo apt-get install openscad")
        
        # Create OpenSCAD file that imports the STL
        scad_content = f"""
        // CSG-style rendering of SDF
        $fn = {config.resolution};
        
        // Import the mesh with proper scaling and orientation
        translate([0, 0, 0])
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
        
        # Run OpenSCAD and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"OpenSCAD error: {result.stderr}")
        
        print(f"Rendered CSG image with OpenSCAD: {output_path}")
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Error during CSG rendering: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(stl_path):
            os.unlink(stl_path)
        if 'scad_path' in locals() and os.path.exists(scad_path):
            os.unlink(scad_path)


def sdf_render_csg_animation(sdf_func: Callable, config: Optional[CSGRenderConfig] = None) -> str:
    """Render a rotating animation of the SDF."""
    if config is None:
        config = CSGRenderConfig()
    
    frames = []
    original_save_path = config.save_path
    
    print(f"Generating {config.n_frames} frames for animation...")
    
    for i in range(config.n_frames):
        angle = 360 * i / config.n_frames
        config.camera_rotation = (angle, 20, 0)
        
        # Create temporary frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            config.save_path = tmp.name
            frame_path = sdf_render_csg(sdf_func, config)
            frames.append(Image.open(frame_path))
            os.unlink(frame_path)
        
        print(f"Rendered frame {i+1}/{config.n_frames}")
    
    # Create GIF
    if original_save_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
    else:
        output_path = original_save_path if original_save_path.endswith('.gif') else original_save_path.replace('.png', '.gif')
    
    print("Creating animated GIF...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//config.fps,
        loop=0
    )
    
    config.save_path = original_save_path
    print(f"Animation saved to: {output_path}")
    
    return output_path