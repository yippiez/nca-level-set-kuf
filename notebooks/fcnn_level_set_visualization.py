# %% [markdown]
# # FCNN Level Set Rendering Demo
# Using the trained FCNN model as an SDF for level set visualization.

# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from util.sdf import sdf_render_level_set, sdf_render_level_set_grid
from util.types import CSGRenderConfig
from util.paths import get_reports_dir
from util.cache import cache_get_torch, cache_get_pickle
from models.fcnn import FCNN

# Create output directory
OUTPUT_DIR = get_reports_dir("fcnn_level_set_visualization")

# %% [markdown]
# ## Load Trained FCNN Model

# %%
# Load model parameters and initialize model
print("Loading trained FCNN model...")
try:
    # Try to load from PyTorch .pth file first
    model_params_path = Path(get_reports_dir('fcnn_basic_v1a1')) / 'sdf_4d_model_params.pth'
    model_params = torch.load(model_params_path)
    print(f"Model parameters loaded from PyTorch file: {model_params}")
except:
    # Fall back to pickle if .pth doesn't exist
    model_params = cache_get_pickle('sdf_4d_model_params')
    print(f"Model parameters loaded from cache: {model_params}")

# Initialize model with loaded parameters
model = FCNN(
    input_size=model_params['input_size'],
    hidden_size=model_params['hidden_size'],
    output_size=model_params['output_size'],
    num_layers=model_params['num_layers']
)

# Load trained weights
try:
    # Try to load from PyTorch .pth file first
    model_weights_path = Path(get_reports_dir('fcnn_basic_v1a1')) / 'sdf_4d_model.pth'
    model_state = torch.load(model_weights_path)
    print(f"Model weights loaded from PyTorch file")
except:
    # Fall back to cache if .pth doesn't exist
    model_state = cache_get_torch('sdf_4d_model')
    print(f"Model weights loaded from cache")

model.load_state_dict(model_state)
model.eval()

# Move to CPU for numpy compatibility
device = torch.device('cpu')
model = model.to(device)
print("Model loaded successfully!")

# %% [markdown]
# ## Create 4D SDF Function Using Trained Model

# %%
def fcnn_sdf(points):
    """FCNN-based 4D SDF function compatible with level set rendering.
    
    Args:
        points: numpy array of shape (N, 4) where columns are [x, y, z, shape]
                shape parameter interpolates between different SDF shapes:
                0=pill, 1=cylinder, 2=box, 3=torus
    
    Returns:
        numpy array of shape (N,) containing SDF values
    """
    # Ensure points is the right shape
    if points.shape[1] != 4:
        raise ValueError(f"Expected points to have 4 columns, got {points.shape[1]}")
    
    # For extrapolation beyond training range, apply cyclic pattern
    shape_values = points[:, 3].copy()
    # Apply modulo to keep shape values in a reasonable range
    # but add a smooth transition
    shape_values_mod = shape_values % 4.0
    
    # Create modified points with cycled shape values
    points_modified = points.copy()
    points_modified[:, 3] = shape_values_mod
    
    # Convert to torch tensor
    points_tensor = torch.FloatTensor(points_modified).to(device)
    
    # Get predictions from model
    with torch.no_grad():
        sdf_values = model(points_tensor).cpu().numpy()
    
    # Flatten to 1D array and clamp values to reasonable range
    sdf_values = sdf_values.flatten()
    # Clamp to prevent marching cubes errors - use tighter bounds for extrapolation
    sdf_values = np.clip(sdf_values, -1.5, 1.5)
    
    return sdf_values

# %% [markdown]
# ## Test the FCNN SDF Function

# %%
# Test with a few sample points
test_points = np.array([
    [0.0, 0.0, 0.0, 0.0],  # Center, pill shape
    [0.0, 0.0, 0.0, 1.0],  # Center, cylinder shape
    [0.0, 0.0, 0.0, 2.0],  # Center, box shape
    [0.0, 0.0, 0.0, 3.0],  # Center, torus shape
    [1.0, 1.0, 1.0, 2.0],  # Outside, box shape
])

test_results = fcnn_sdf(test_points)
print("Test SDF values:")
for i, (point, value) in enumerate(zip(test_points, test_results)):
    shape_names = ['pill', 'cylinder', 'box', 'torus']
    shape_idx = int(point[3])
    shape_name = shape_names[shape_idx] if 0 <= shape_idx <= 3 else f'shape={point[3]}'
    print(f"  Point {point[:3]} ({shape_name}): SDF = {value:.4f}")

# %% [markdown]
# ## Level Set Animation - Morphing Between Shapes

# %%
# Create animation showing morphing between different shapes
config_morph = CSGRenderConfig(
    grid_size=50,  # Higher resolution than quick test
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "fcnn_morph_animation.gif"),
    image_size=(600, 600),
    camera_rotation=(45, 20, 0),
    n_frames=30,  # Smooth animation
    fps=10,
    colorscheme="Cornfield"
)

# Animate morphing from pill (0) to torus (3)
print("Rendering morphing animation...")
shape_values_morph = np.linspace(0.0, 3.0, config_morph.n_frames)
gif_path = sdf_render_level_set(fcnn_sdf, config_morph, shape_values=shape_values_morph)
print(f"Saved animation to: {gif_path}")

# %% [markdown]
# ## Grid Visualization - Different Shape Values

# %%
# Create grid showing different shape values
config_grid = CSGRenderConfig(
    grid_size=50,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "fcnn_shape_grid.png"),
    image_size=(400, 400),
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield"
)

# Show all four shapes plus some intermediate values, but stay within trained range
shape_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
print("Rendering shape grid...")
grid_path = sdf_render_level_set_grid(fcnn_sdf, config_grid, shape_values)
print(f"Saved grid to: {grid_path}")

# %% [markdown]
# ## Custom Animation - Extended Range from 0 to 6 with Error Handling

# %%
import tempfile
from PIL import Image, ImageDraw, ImageFont

def sdf_render_level_set_safe(sdf_func, config, shape_values):
    """Safe version of sdf_render_level_set that handles marching cubes failures."""
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
            
            try:
                # Try to render normally
                from util.sdf import sdf_render_csg
                frame_path = sdf_render_csg(current_sdf, config)
                
                # Load image and add text annotation
                img = Image.open(frame_path)
            except ValueError as e:
                if "Surface level must be within volume data range" in str(e):
                    # Create error placeholder image
                    img = Image.new('RGB', config.image_size, color='black')
                    draw = ImageDraw.Draw(img)
                    
                    # Add error text
                    text = "Marching Cube Failed"
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
                    except:
                        font = ImageFont.load_default()
                    
                    # Center the text
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    position = ((config.image_size[0] - text_width) // 2, 
                               (config.image_size[1] - text_height) // 2)
                    draw.text(position, text, fill='red', font=font)
                    
                    print(f"Frame {i+1}/{len(shape_values)} (shape={shape_val:.2f}) - Marching cubes failed")
                else:
                    raise  # Re-raise other errors
            
            # Add shape parameter text
            draw = ImageDraw.Draw(img)
            text = f"shape={shape_val:.2f}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Top right position
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            position = (config.image_size[0] - text_width - 10, 10)
            draw.text(position, text, fill='white', font=font)
            
            # Save frame
            img.save(tmp.name)
            frames.append(Image.open(tmp.name))
            
            if (i+1) % 10 == 0 or i == len(shape_values) - 1:
                print(f"Rendered frame {i+1}/{len(shape_values)} (shape={shape_val:.2f})")
    
    # Create GIF
    print("Creating animated GIF...")
    output_path = original_save_path
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//config.fps,
        loop=0
    )
    
    print(f"Level set animation saved to: {output_path}")
    return output_path

# Create a custom animation with extended range from 0 to 6
config_custom = CSGRenderConfig(
    grid_size=50,
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "fcnn_extended_transition.gif"),
    image_size=(600, 600),
    camera_rotation=(30, 45, 0),  # Different camera angle
    n_frames=60,  # Many frames for slower transition
    fps=10,  # 6 seconds total animation
    colorscheme="Metallic"  # Grey/metallic color scheme (now default)
)

# Animate from 0 to 6, with error handling for marching cubes failures
print("Rendering extended transition animation (0 to 6)...")
shape_values_custom = np.linspace(0.0, 6.0, config_custom.n_frames)
gif_path_custom = sdf_render_level_set_safe(fcnn_sdf, config_custom, shape_values_custom)
print(f"Saved custom animation to: {gif_path_custom}")

# %% [markdown]
# ## Compare Different Grid Resolutions

# %%
# Low resolution grid for comparison
config_low_res = CSGRenderConfig(
    grid_size=30,  # Lower resolution
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "fcnn_grid_low_res.png"),
    image_size=(300, 300),
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield"
)

shape_values_simple = [0.0, 1.0, 2.0, 3.0]
print("Rendering low resolution grid...")
grid_path_low = sdf_render_level_set_grid(fcnn_sdf, config_low_res, shape_values_simple)
print(f"Saved low res grid to: {grid_path_low}")

# High resolution grid
config_high_res = CSGRenderConfig(
    grid_size=80,  # Higher resolution
    bounds=(-1.5, 1.5),
    save_path=str(OUTPUT_DIR / "fcnn_grid_high_res.png"),
    image_size=(500, 500),
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield"
)

print("Rendering high resolution grid...")
grid_path_high = sdf_render_level_set_grid(fcnn_sdf, config_high_res, shape_values_simple)
print(f"Saved high res grid to: {grid_path_high}")

# %% [markdown]
# ## Done!
# 
# The FCNN model has been successfully used as an SDF for level set rendering.
# Check the `reports/fcnn_level_set_visualization/` directory for all generated visualizations.