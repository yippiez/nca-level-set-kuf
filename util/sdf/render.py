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
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:
                normal = normal / norm_length
            else:
                normal = np.array([0, 0, 1])  # Default normal if degenerate
            
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


def sdf_render_level_set(sdf_func: Callable, config: Optional[CSGRenderConfig] = None, shape_values: Optional[list] = None) -> str:
    """Render a morphing animation iterating through shape parameter values."""
    if config is None:
        config = CSGRenderConfig()
    
    if shape_values is None:
        shape_values = np.linspace(0, 4, config.n_frames)
    
    frames = []
    original_save_path = config.save_path
    
    print(f"Generating {len(shape_values)} frames for level set animation...")
    
    for i, shape_val in enumerate(shape_values):
        # Create SDF function with current shape value
        def current_sdf(points):
            # Add shape parameter as 4th dimension
            points_4d = np.concatenate([points, np.full((points.shape[0], 1), shape_val)], axis=1)
            return sdf_func(points_4d)
        
        # Create temporary frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            config.save_path = tmp.name
            frame_path = sdf_render_csg(current_sdf, config)
            
            # Load image and add text annotation
            img = Image.open(frame_path)
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Add shape parameter text in top right
            text = f"shape={shape_val:.2f}"
            try:
                # Try to use a better font if available
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position (top right)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_position = (img.width - text_width - 20, 20)
            
            # Draw text with background
            draw.rectangle([text_position[0] - 5, text_position[1] - 5,
                          text_position[0] + text_width + 5, text_position[1] + text_height + 5],
                          fill='white', outline='black')
            draw.text(text_position, text, fill='black', font=font)
            
            frames.append(img)
            os.unlink(frame_path)
        
        print(f"Rendered frame {i+1}/{len(shape_values)} (shape={shape_val:.2f})")
    
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
    print(f"Level set animation saved to: {output_path}")
    
    return output_path


def sdf_render_level_set_grid(sdf_func: Callable, config: Optional[CSGRenderConfig] = None, shape_values: Optional[list] = None) -> str:
    """Render a grid of images with different shape parameter values."""
    if config is None:
        config = CSGRenderConfig()
    
    if shape_values is None:
        shape_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    # Calculate grid dimensions
    n_images = len(shape_values)
    grid_cols = int(np.ceil(np.sqrt(n_images)))
    grid_rows = int(np.ceil(n_images / grid_cols))
    
    # Create individual images
    images = []
    original_save_path = config.save_path
    
    print(f"Generating {n_images} images for level set grid...")
    
    for i, shape_val in enumerate(shape_values):
        # Create SDF function with current shape value
        def current_sdf(points):
            # Add shape parameter as 4th dimension
            points_4d = np.concatenate([points, np.full((points.shape[0], 1), shape_val)], axis=1)
            return sdf_func(points_4d)
        
        # Create temporary frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            config.save_path = tmp.name
            frame_path = sdf_render_csg(current_sdf, config)
            
            # Load image and add text annotation
            img = Image.open(frame_path)
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Add shape parameter text
            text = f"s={shape_val:.2f}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position (centered at bottom)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_position = ((img.width - text_width) // 2, img.height - text_height - 30)
            
            # Draw text with background
            draw.rectangle([text_position[0] - 10, text_position[1] - 5,
                          text_position[0] + text_width + 10, text_position[1] + text_height + 5],
                          fill='white', outline='black')
            draw.text(text_position, text, fill='black', font=font)
            
            images.append(img)
            os.unlink(frame_path)
        
        print(f"Rendered image {i+1}/{n_images} (s={shape_val:.2f})")
    
    # Create grid image
    grid_width = grid_cols * config.image_size[0]
    grid_height = grid_rows * config.image_size[1]
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
    
    for i, img in enumerate(images):
        row = i // grid_cols
        col = i % grid_cols
        x = col * config.image_size[0]
        y = row * config.image_size[1]
        grid_img.paste(img, (x, y))
    
    # Save grid image
    if original_save_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    else:
        output_path = original_save_path
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_path)
    
    config.save_path = original_save_path
    print(f"Level set grid saved to: {output_path}")
    
    return output_path