# %% [markdown]
# # FCNN Range Models Training and Visualization
# 
# This notebook trains multiple FCNN models with different shape parameter ranges:
# - Model 1: range [0, 6] with 15 shapes
# - Model 2: range [0, 4] with 15 shapes  
# - Model 3: range [0, 2] with 15 shapes
# - Model 4: range [0, 1] with 15 shapes
# - Model 5: range [0, 0.5] with 15 shapes
#
# Each model is named `mnb_core_0-<n>_<m>` where:
# - n: upper bound of the range
# - m: number of shapes (15 for all models)
#
# The models learn to predict SDFs for all available shapes:
# - shape=0: Sphere
# - shape=1: Pill
# - shape=2: Box
# - shape=3: Torus
# - shape=4: Cylinder
# - shape=5: Cone
# - shape=6: Octahedron
# - shape=7: Pyramid
# - shape=8: Hexagonal Prism
# - shape=9: Ellipsoid
# - shape=10: Rounded Box
# - shape=11: Link
# - shape=12: Star
#
# For each trained model, this notebook also generates:
# 1. Grid visualizations - a grid of models at different shape parameter values
# 2. Animation visualizations - morphing animations between different shape values

# %%
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Any, Optional, Callable
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from models.fcnn import FCNN
from util.paths import get_reports_dir
from util.sdf import (sdf_sphere, sdf_pill, sdf_box, sdf_torus, sdf_cylinder,
                      sdf_cone, sdf_octahedron, sdf_pyramid, sdf_hexagonal_prism,
                      sdf_ellipsoid, sdf_rounded_box, sdf_link, sdf_star)
# Cache imports removed in refactoring
from util.types import CSGRenderConfig
from util.sdf.render import (sdf_render_csg, sdf_render_csg_animation, 
                           sdf_render_level_set, sdf_render_level_set_grid,
                           sdf_render_level_set_side_to_side)
from util.sdf.similarity import compare_sdf, compare_sdf_batch

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## Dataset Generation Function

# %%
def generate_range_dataset(num_points_per_shape: int, shape_range: Tuple[float, float], num_shapes: int = 13):
    """Generate dataset with shape parameters mapped to specified range.
    
    Args:
        num_points_per_shape: Number of 3D points to sample per shape
        shape_range: Tuple (min, max) for shape parameter range
        num_shapes: Number of distinct shapes (13 available shapes)
    
    Returns:
        inputs: Tensor of shape (N, 4) with columns [x, y, z, shape]
        targets: Tensor of shape (N, 1) with SDF values
    """
    # Map discrete shape indices to continuous range
    shape_min, shape_max = shape_range
    shape_values = np.linspace(shape_min, shape_max, num_shapes)
    
    # Define bounds for 3D point sampling
    bounds = (-1.5, 1.5)
    
    all_inputs = []
    all_targets = []
    
    # Generate points for each shape
    for shape_idx in range(num_shapes):
        points = np.random.uniform(bounds[0], bounds[1], (num_points_per_shape, 3))
        
        # Calculate SDF based on original shape index
        if shape_idx == 0:  # Sphere
            center = np.array([0, 0, 0])
            radius = 0.5
            distances = sdf_sphere(points, center, radius)
        elif shape_idx == 1:  # Pill
            p1 = np.array([-0.5, 0, 0])
            p2 = np.array([0.5, 0, 0])
            radius = 0.3
            distances = sdf_pill(points, p1, p2, radius)
        elif shape_idx == 2:  # Box
            center = np.array([0, 0, 0])
            dimensions = np.array([0.8, 0.8, 0.8])
            distances = sdf_box(points, center, dims=dimensions)
        elif shape_idx == 3:  # Torus
            center = np.array([0, 0, 0])
            major_radius = 0.5
            minor_radius = 0.2
            distances = sdf_torus(points, center, r_major=major_radius, r_minor=minor_radius)
        elif shape_idx == 4:  # Cylinder
            center = np.array([0, 0, 0])
            radius = 0.3
            height = 1.0
            distances = sdf_cylinder(points, center, radius, height)
        elif shape_idx == 5:  # Cone
            tip = np.array([0, 0.5, 0])
            base_center = np.array([0, -0.5, 0])
            radius = 0.5
            distances = sdf_cone(points, tip, base_center, radius)
        elif shape_idx == 6:  # Octahedron
            center = np.array([0, 0, 0])
            radius = 0.7
            distances = sdf_octahedron(points, center, radius)
        elif shape_idx == 7:  # Pyramid
            tip = np.array([0, 0.5, 0])
            base_center = np.array([0, -0.5, 0])
            base_size = 0.8
            distances = sdf_pyramid(points, tip, base_center, base_size)
        elif shape_idx == 8:  # Hexagonal Prism
            center = np.array([0, 0, 0])
            radius = 0.5
            height = 0.8
            distances = sdf_hexagonal_prism(points, center, radius, height)
        elif shape_idx == 9:  # Ellipsoid
            center = np.array([0, 0, 0])
            radii = np.array([0.7, 0.4, 0.5])
            distances = sdf_ellipsoid(points, center, radii)
        elif shape_idx == 10:  # Rounded Box
            center = np.array([0, 0, 0])
            dimensions = np.array([0.8, 0.8, 0.8])
            radius = 0.1
            distances = sdf_rounded_box(points, center, dimensions, radius)
        elif shape_idx == 11:  # Link
            center = np.array([0, 0, 0])
            r_major = 0.3
            r_minor = 0.1
            length = 0.6
            distances = sdf_link(points, center, r_major, r_minor, length)
        else:  # shape_idx == 12, Star
            center = np.array([0, 0, 0])
            n_points = 5
            r_outer = 0.7
            r_inner = 0.3
            thickness = 0.2
            distances = sdf_star(points, center, n_points, r_outer, r_inner, thickness)
        
        # Use mapped shape value instead of discrete index
        shape_param = np.full((num_points_per_shape, 1), shape_values[shape_idx])
        inputs = np.hstack([points, shape_param])
        
        all_inputs.append(inputs)
        all_targets.append(distances.reshape(-1, 1))
    
    # Combine all data
    inputs = np.vstack(all_inputs)
    targets = np.vstack(all_targets)
    
    # Convert to tensors
    return torch.FloatTensor(inputs), torch.FloatTensor(targets)

# %% [markdown]
# ## Model Training Function

# %%
class SDFDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    def __len__(self):
        return len(self.inputs)

def train_range_model(shape_range: Tuple[float, float], num_shapes: int = 13, 
                     num_points: int = 100000, num_epochs: int = 50):
    """Train an FCNN model for the specified shape range.
    
    Args:
        shape_range: Tuple (min, max) for shape parameter range
        num_shapes: Number of shapes to include
        num_points: Total number of training points
        num_epochs: Number of training epochs
    
    Returns:
        Trained model
    """
    # Generate model name
    range_min, range_max = shape_range
    model_name = f"mnb_core_0-{range_max}_{num_shapes}"
    
    # Check if model already exists in reports directory
    reports_dir = get_reports_dir('fcnn_range_models')
    model_save_path = os.path.join(reports_dir, f'{model_name}.pt')
    
    try:
        if os.path.exists(model_save_path):
            print(f"Loading existing model: {model_name}")
            checkpoint = torch.load(model_save_path, map_location=device)
            model_params = checkpoint['model_params']
            model = FCNN(**model_params)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            return model, model_name
    except (FileNotFoundError, Exception) as e:
        print(f"No existing model found or error loading model: {e}")
        # Continue to train a new model
    
    print(f"Training new model: {model_name}")
    print(f"Shape range: [{range_min}, {range_max}]")
    
    # Generate dataset
    num_points_per_shape = num_points // num_shapes
    inputs, targets = generate_range_dataset(num_points_per_shape, shape_range, num_shapes)
    
    # Split into train/validation
    num_samples = len(inputs)
    indices = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    
    train_inputs = inputs[indices[:train_size]]
    train_targets = targets[indices[:train_size]]
    val_inputs = inputs[indices[train_size:]]
    val_targets = targets[indices[train_size:]]
    
    # Create dataloaders
    train_dataset = SDFDataset(train_inputs, train_targets)
    val_dataset = SDFDataset(val_inputs, val_targets)
    
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model_params = {
        'input_size': 4,
        'hidden_size': 64,
        'output_size': 1,
        'num_layers': 5
    }
    model = FCNN(**model_params).to(device)
    
    # Train
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Save model in PyTorch format
    reports_dir = get_reports_dir('fcnn_range_models')
    model_save_path = os.path.join(reports_dir, f'{model_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params,
        'shape_range': shape_range,
        'num_shapes': num_shapes
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    return model, model_name


# %% [markdown]
# ## Main Training and Visualization Loop

# %%
# Define shape ranges to train
shape_ranges = [
    (0, 6),
    (0, 4),
    (0, 2),
    (0, 1),
    (0, 0.5)
]

# Train models
trained_models = []

# Train models
trained_models = []

for shape_range in shape_ranges:
    print(f"\n{'='*50}")
    print(f"Processing range: {shape_range}")
    print(f"{'='*50}")
    
    # Train model
    model, model_name = train_range_model(
        shape_range=shape_range,
        num_shapes=13,     # Use all available shapes
        num_points=100000, # Full training data
        num_epochs=50      # Full training epochs
    )
    
    trained_models.append((model, model_name, shape_range))
    print(f"Completed training for range {shape_range}")

print("\n\nAll models trained!")
print("Models saved:")
for _, model_name, shape_range in trained_models:
    print(f"  - {model_name}: range {shape_range}")

# %% [markdown]
# ## Generate Visualizations for Each Model
# 
# For each trained model, we'll create:
# 1. Grid visualizations - showing different shape parameter values
# 2. Animation visualizations - morphing between different shape values

# %%
def create_model_sdf_function(model: FCNN, shape_param: float) -> Callable:
    """Create an SDF function for a specific shape parameter value.
    
    Args:
        model: The trained FCNN model
        shape_param: The shape parameter value
        
    Returns:
        A function that takes 3D points and returns SDF values
    """
    def sdf_func(points):
        # Convert to tensor
        if not isinstance(points, torch.Tensor):
            points = torch.FloatTensor(points)
        
        # Reshape if needed
        original_shape = points.shape
        if len(original_shape) > 2:
            points = points.reshape(-1, original_shape[-1])
        
        # Add shape parameter
        batch_size = points.shape[0]
        shape_params = torch.full((batch_size, 1), shape_param, dtype=torch.float32)
        model_inputs = torch.cat([points, shape_params], dim=1)
        
        # Move to device
        model_inputs = model_inputs.to(device)
        
        # Generate prediction
        with torch.no_grad():
            sdf_values = model(model_inputs).cpu().numpy()
        
        # Reshape back to original
        if len(original_shape) > 2:
            sdf_values = sdf_values.reshape(original_shape[:-1])
        
        return sdf_values
    
    return sdf_func

def create_model_level_set_function(model: FCNN) -> Callable:
    """Create a 4D level set function for the model.
    
    Args:
        model: The trained FCNN model
        
    Returns:
        A function that takes 4D points (x,y,z,shape) and returns SDF values
    """
    def level_set_func(points_4d):
        # Convert to tensor
        if not isinstance(points_4d, torch.Tensor):
            points_4d = torch.FloatTensor(points_4d)
        
        # Reshape if needed
        original_shape = points_4d.shape
        if len(original_shape) > 2:
            points_4d = points_4d.reshape(-1, original_shape[-1])
        
        # Move to device
        points_4d = points_4d.to(device)
        
        # Generate prediction
        with torch.no_grad():
            sdf_values = model(points_4d).cpu().numpy()
        
        # Reshape back to original
        if len(original_shape) > 2:
            sdf_values = sdf_values.reshape(original_shape[:-1])
        
        return sdf_values
    
    return level_set_func

def generate_visualizations(model: FCNN, model_name: str, shape_range: Tuple[float, float], num_shapes: int = 13):
    """Generate grid and animation visualizations for a model.
    
    Args:
        model: The trained FCNN model
        model_name: Name of the model
        shape_range: Tuple of (min, max) shape parameter range
        num_shapes: Number of shapes in the model
    """
    # Set up the reports directory for this model
    reports_dir = get_reports_dir('fcnn_range_models')
    
    # Create render configuration
    config = CSGRenderConfig(
        grid_size=50,  # Reduced for faster rendering
        resolution=50,  # Reduced for faster rendering
        image_size=(400, 400),
        n_frames=30,  # Reduced for faster animation
        fps=10
    )
    
    # Prepare shape values for grid visualization
    shape_min, shape_max = shape_range
    
    # 1. Generate grid visualization
    grid_filename = f"{model_name}_grid.png"
    grid_path = os.path.join(reports_dir, grid_filename)
    
    # Skip if file already exists
    if os.path.exists(grid_path):
        print(f"Grid visualization already exists at {grid_path}, skipping...")
    else:
        print(f"Generating grid visualization for {model_name}...")
        
        # Create shape values for grid
        grid_shape_values = np.linspace(shape_min, shape_max, min(num_shapes, 9))
        
        # Create level set function
        level_set_func = create_model_level_set_function(model)
        
        # Set save path and render
        config.save_path = grid_path
        sdf_render_level_set_grid(level_set_func, config, shape_values=grid_shape_values)
    
    # 2. Generate animation visualization
    anim_filename = f"{model_name}_animation.gif"
    anim_path = os.path.join(reports_dir, anim_filename)
    
    # Skip if file already exists
    if os.path.exists(anim_path):
        print(f"Animation visualization already exists at {anim_path}, skipping...")
    else:
        print(f"Generating animation visualization for {model_name}...")
        
        # Create shape values for animation
        anim_shape_values = np.linspace(shape_min, shape_max, config.n_frames)
        
        # Create level set function
        level_set_func = create_model_level_set_function(model)
        
        # Set save path and render
        config.save_path = anim_path
        sdf_render_level_set(level_set_func, config, shape_values=anim_shape_values)
    
    return grid_path, anim_path

# %%
# Generate visualizations for all trained models
visualization_paths = []

print("\n\nGenerating visualizations for all models:")
for model, model_name, shape_range in trained_models:
    print(f"\n{'='*50}")
    print(f"Generating visualizations for {model_name} (range {shape_range})")
    print(f"{'='*50}")
    
    grid_path, anim_path = generate_visualizations(model, model_name, shape_range)
    visualization_paths.append((model_name, grid_path, anim_path))

print("\n\nAll visualizations generated!")
print("Visualization paths:")
for model_name, grid_path, anim_path in visualization_paths:
    print(f"  - {model_name}:")
    print(f"    - Grid: {grid_path}")
    print(f"    - Animation: {anim_path}")

# %% [markdown]
# ## Generate Side-by-Side Comparisons with Ground Truth
# 
# For each trained model, we'll create side-by-side comparisons with ground truth shapes to visually 
# assess how well the model learned different shapes across its parameter range.
# 
# We'll also calculate similarity metrics between the learned and ground truth shapes.

# %%
def create_ground_truth_sdf(shape_idx, shape_param_value):
    """Create ground truth SDF function for a specific shape and parameter value.
    
    Args:
        shape_idx: Index of the shape type (0-12)
        shape_param_value: Shape parameter value
        
    Returns:
        SDF function for the specified shape
    """
    def ground_truth_sdf(points):
        # Reshape if needed
        original_shape = points.shape
        if len(original_shape) > 2:
            points = points.reshape(-1, original_shape[-1])
        
        # Calculate SDF based on shape index
        if shape_idx == 0:  # Sphere
            center = np.array([0, 0, 0])
            radius = 0.5
            distances = sdf_sphere(points, center, radius)
        elif shape_idx == 1:  # Pill
            p1 = np.array([-0.5, 0, 0])
            p2 = np.array([0.5, 0, 0])
            radius = 0.3
            distances = sdf_pill(points, p1, p2, radius)
        elif shape_idx == 2:  # Box
            center = np.array([0, 0, 0])
            dimensions = np.array([0.8, 0.8, 0.8])
            distances = sdf_box(points, center, dims=dimensions)
        elif shape_idx == 3:  # Torus
            center = np.array([0, 0, 0])
            major_radius = 0.5
            minor_radius = 0.2
            distances = sdf_torus(points, center, r_major=major_radius, r_minor=minor_radius)
        elif shape_idx == 4:  # Cylinder
            center = np.array([0, 0, 0])
            radius = 0.3
            height = 1.0
            distances = sdf_cylinder(points, center, radius, height)
        elif shape_idx == 5:  # Cone
            tip = np.array([0, 0.5, 0])
            base_center = np.array([0, -0.5, 0])
            radius = 0.5
            distances = sdf_cone(points, tip, base_center, radius)
        elif shape_idx == 6:  # Octahedron
            center = np.array([0, 0, 0])
            radius = 0.7
            distances = sdf_octahedron(points, center, radius)
        elif shape_idx == 7:  # Pyramid
            tip = np.array([0, 0.5, 0])
            base_center = np.array([0, -0.5, 0])
            base_size = 0.8
            distances = sdf_pyramid(points, tip, base_center, base_size)
        elif shape_idx == 8:  # Hexagonal Prism
            center = np.array([0, 0, 0])
            radius = 0.5
            height = 0.8
            distances = sdf_hexagonal_prism(points, center, radius, height)
        elif shape_idx == 9:  # Ellipsoid
            center = np.array([0, 0, 0])
            radii = np.array([0.7, 0.4, 0.5])
            distances = sdf_ellipsoid(points, center, radii)
        elif shape_idx == 10:  # Rounded Box
            center = np.array([0, 0, 0])
            dimensions = np.array([0.8, 0.8, 0.8])
            radius = 0.1
            distances = sdf_rounded_box(points, center, dimensions, radius)
        elif shape_idx == 11:  # Link
            center = np.array([0, 0, 0])
            r_major = 0.3
            r_minor = 0.1
            length = 0.6
            distances = sdf_link(points, center, r_major, r_minor, length)
        else:  # shape_idx == 12, Star
            center = np.array([0, 0, 0])
            n_points = 5
            r_outer = 0.7
            r_inner = 0.3
            thickness = 0.2
            distances = sdf_star(points, center, n_points, r_outer, r_inner, thickness)
        
        # Reshape back to original if needed
        if len(original_shape) > 2:
            distances = distances.reshape(original_shape[:-1])
            
        return distances
    
    return ground_truth_sdf

def generate_side_by_side_comparison(model, model_name, shape_range):
    """Generate side-by-side comparison between learned model and ground truth.
    
    Args:
        model: Trained FCNN model
        model_name: Name of the model
        shape_range: Shape parameter range as (min, max)
    """
    # Set up reports directory
    reports_dir = get_reports_dir('fcnn_range_models')
    
    # Create rendering configuration
    config = CSGRenderConfig(
        grid_size=50,  # Reduced for faster rendering
        resolution=50,  # Reduced for faster rendering
        image_size=(400, 400)
    )
    
    # Create output path
    comparison_filename = f"{model_name}_comparison.png"
    comparison_path = os.path.join(reports_dir, comparison_filename)
    
    # Skip if file already exists
    if os.path.exists(comparison_path):
        print(f"Side-by-side comparison already exists at {comparison_path}, skipping...")
        return comparison_path
    
    # Create level set function from model
    model_level_set_func = create_model_level_set_function(model)
    
    # Create shape indices and parameter values
    shape_min, shape_max = shape_range
    num_shapes = 13  # Number of distinct shapes
    
    # Map discrete shape indices to continuous range
    shape_values = np.linspace(shape_min, shape_max, num_shapes)
    
    # Create target SDF functions
    target_sdfs = []
    for i in range(num_shapes):
        target_sdfs.append(create_ground_truth_sdf(i, shape_values[i]))
    
    # Calculate similarity metrics
    print(f"Calculating similarity metrics for {model_name}...")
    iou_values = compare_sdf_batch(model_level_set_func, target_sdfs, shape_values, method='iou')
    mse_values = compare_sdf_batch(model_level_set_func, target_sdfs, shape_values, method='mse')
    
    # Prepare metrics for display
    metrics = [
        ('IoU', iou_values),
        ('MSE', mse_values)
    ]
    
    # Generate side-by-side comparison
    print(f"Generating side-by-side comparison for {model_name}...")
    config.save_path = comparison_path
    sdf_render_level_set_side_to_side(
        model_level_set_func, target_sdfs, shape_values, config, metrics
    )
    
    return comparison_path

# %%
# Generate side-by-side comparisons for all models
comparison_paths = []

print("\n\nGenerating side-by-side comparisons:")
for model, model_name, shape_range in trained_models:
    print(f"\n{'='*50}")
    print(f"Generating comparison for {model_name} (range {shape_range})")
    print(f"{'='*50}")
    
    comparison_path = generate_side_by_side_comparison(model, model_name, shape_range)
    comparison_paths.append((model_name, comparison_path))

print("\n\nAll comparisons generated!")
print("Comparison paths:")
for model_name, path in comparison_paths:
    print(f"  - {model_name}: {path}")