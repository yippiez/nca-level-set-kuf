# %% [markdown]
# # FCNN SDF Learning Experiment
# 
# This experiment trains a fully connected neural network (FCNN) to learn different signed distance functions (SDFs).
# 
# The model learns to predict the distance for:
# - s=0: Pill shape
# - s=1: Cylinder 
# - s=2: Box
# - s=3: Torus
# 
# **Input to the model:** [x, y, z, shape_type]  
# **Output from the model:** distance value

# %%
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FCNN model and SDF functions
from models.fcnn import FCNN
from util.paths import get_reports_dir
from util.sdf import sdf_pill, sdf_box, sdf_torus, sdf_render
from util.cache import cache_save, cache_exists, cache_get_torch, cache_get_pickle, cache_get_numpy

# %% 
# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## SDF Function Definitions
# 
# We implement a cylinder SDF function since it's not included in the util.sdf module.

# %%
# %% [markdown]
# ## SDF Function Definitions
# 
# We'll use the SDF functions from the utility module.
# For cylinder shape, we can adapt the pill function by using appropriate parameters.

# %% [markdown]
# ## Data Generation
# 
# We generate synthetic data for training the model. This is a computationally expensive operation,
# so we cache the results to avoid regenerating the data each time.

# %%
def generate_data(num_points_per_shape=10000):
    """
    Generate synthetic training data for different SDF shapes.
    
    Args:
        num_points_per_shape: Number of random points to generate per shape
        
    Returns:
        Tuple of (inputs, targets) as PyTorch tensors
    """
    # Check if data is already cached
    cache_name = f"sdf_data_{num_points_per_shape}"
    if cache_exists(cache_name, extension="pt"):
        print(f"Loading cached SDF data from {cache_name}")
        data_dict = cache_get_torch(cache_name)
        return data_dict['inputs'], data_dict['targets']
    
    print(f"Generating {num_points_per_shape} points per shape...")
    
    # Define the bounds for random point generation
    bounds = (-1.5, 1.5)
    
    # Lists to store inputs and targets
    all_inputs = []
    all_targets = []
    
    # Generate random points in 3D space
    points = np.random.uniform(bounds[0], bounds[1], (num_points_per_shape, 3))
    
    # Shape 0: Pill
    p1 = np.array([-0.5, 0, 0])
    p2 = np.array([0.5, 0, 0])
    radius = 0.3
    distances_pill = sdf_pill(points, p1, p2, radius)
    shape_indices_pill = np.zeros((num_points_per_shape, 1))
    inputs_pill = np.hstack([points, shape_indices_pill])
    all_inputs.append(inputs_pill)
    all_targets.append(distances_pill.reshape(-1, 1))
      # Shape 1: Cylinder (using pill function with endpoints at top and bottom)
    center = np.array([0, 0, 0])
    radius = 0.3
    height = 1.0
    p1_cyl = center - np.array([0, height/2, 0])  # Bottom center
    p2_cyl = center + np.array([0, height/2, 0])  # Top center
    distances_cylinder = sdf_pill(points, p1_cyl, p2_cyl, radius)
    shape_indices_cylinder = np.ones((num_points_per_shape, 1))
    inputs_cylinder = np.hstack([points, shape_indices_cylinder])
    all_inputs.append(inputs_cylinder)
    all_targets.append(distances_cylinder.reshape(-1, 1))
    
    # Shape 2: Box
    center = np.array([0, 0, 0])
    dimensions = np.array([0.8, 0.8, 0.8])
    distances_box = sdf_box(points, center, dims=dimensions)
    shape_indices_box = np.ones((num_points_per_shape, 1)) * 2
    inputs_box = np.hstack([points, shape_indices_box])
    all_inputs.append(inputs_box)
    all_targets.append(distances_box.reshape(-1, 1))
    
    # Shape 3: Torus
    center = np.array([0, 0, 0])
    major_radius = 0.5
    minor_radius = 0.2
    distances_torus = sdf_torus(points, center, r_major=major_radius, r_minor=minor_radius)
    shape_indices_torus = np.ones((num_points_per_shape, 1)) * 3
    inputs_torus = np.hstack([points, shape_indices_torus])
    all_inputs.append(inputs_torus)
    all_targets.append(distances_torus.reshape(-1, 1))
      # Combine all data
    inputs = np.vstack(all_inputs)
    targets = np.vstack(all_targets)
    
    # Convert to PyTorch tensors
    inputs_tensor = torch.FloatTensor(inputs)
    targets_tensor = torch.FloatTensor(targets)
    
    # Cache the data for future use
    data_dict = {
        'inputs': inputs_tensor,
        'targets': targets_tensor
    }
    cache_save(data_dict, cache_name)
    print(f"SDF data cached as {cache_name}")
    
    return inputs_tensor, targets_tensor

# %% [markdown]
# ## Dataset and DataLoader Classes
#
# We define a PyTorch Dataset class to handle our synthetic SDF data.

# %%
class SDFDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    def __len__(self):
        return len(self.inputs)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model and validate it.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        
    Returns:
        Lists of training and validation losses
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

def visualize_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_predictions(model, shape_type, grid_resolution=30):
    """
    Visualize the predictions of the model for a specific shape type.
    
    Args:
        model: The trained model
        shape_type: Integer (0=pill, 1=cylinder, 2=box, 3=torus)
        grid_resolution: Number of points along each axis in the visualization grid
    """
    model.eval()
    
    # Create a grid of points for visualization
    x = np.linspace(-1.0, 1.0, grid_resolution)
    y = np.linspace(-1.0, 1.0, grid_resolution)
    z = np.linspace(-1.0, 1.0, grid_resolution)
    
    # Slice at z=0 for 2D visualization
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    # Prepare inputs for the model
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
    shape_indices = np.ones((len(points), 1)) * shape_type
    inputs = np.hstack([points, shape_indices])
    
    # Get predictions from the model
    inputs_tensor = torch.FloatTensor(inputs).to(device)
    
    with torch.no_grad():
        predictions = model(inputs_tensor).cpu().numpy()
    
    # Reshape predictions back to grid shape
    predictions_grid = predictions.reshape(grid_resolution, grid_resolution)
      # Calculate ground truth distances for comparison
    if shape_type == 0:  # Pill
        p1 = np.array([-0.5, 0, 0])
        p2 = np.array([0.5, 0, 0])
        radius = 0.3
        ground_truth = sdf_pill(points, p1, p2, radius)
    elif shape_type == 1:  # Cylinder (using pill function)
        center = np.array([0, 0, 0])
        radius = 0.3
        height = 1.0
        p1_cyl = center - np.array([0, height/2, 0])  # Bottom center
        p2_cyl = center + np.array([0, height/2, 0])  # Top center
        ground_truth = sdf_pill(points, p1_cyl, p2_cyl, radius)
    elif shape_type == 2:  # Box
        center = np.array([0, 0, 0])
        dimensions = np.array([0.8, 0.8, 0.8])
        ground_truth = sdf_box(points, center, dims=dimensions)
    elif shape_type == 3:  # Torus
        center = np.array([0, 0, 0])
        major_radius = 0.5
        minor_radius = 0.2
        ground_truth = sdf_torus(points, center, r_major=major_radius, r_minor=minor_radius)
    
    ground_truth_grid = ground_truth.reshape(grid_resolution, grid_resolution)
    
    # Create the visualization
    shape_names = ['Pill', 'Cylinder', 'Box', 'Torus']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot ground truth
    im0 = axes[0].contourf(xx, yy, ground_truth_grid, levels=20, cmap='viridis')
    axes[0].contour(xx, yy, ground_truth_grid, levels=[0], colors='red')
    axes[0].set_title(f'Ground Truth - {shape_names[shape_type]}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    
    # Plot predictions
    im1 = axes[1].contourf(xx, yy, predictions_grid, levels=20, cmap='viridis')
    axes[1].contour(xx, yy, predictions_grid, levels=[0], colors='red')
    axes[1].set_title(f'Model Predictions - {shape_names[shape_type]}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_aspect('equal')
    
    # Plot differences
    diff = np.abs(ground_truth_grid - predictions_grid)
    im2 = axes[2].contourf(xx, yy, diff, levels=20, cmap='inferno')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_aspect('equal')
    
    fig.colorbar(im0, ax=axes[0], label='SDF Value')
    fig.colorbar(im1, ax=axes[1], label='SDF Value')
    fig.colorbar(im2, ax=axes[2], label='Absolute Error')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the experiment."""
    # Generate synthetic data
    print("Generating synthetic data...")
    inputs, targets = generate_data(num_points_per_shape=10000)
    
    # Split data into training and validation sets (80/20 split)
    dataset_size = len(inputs)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_inputs, train_targets = inputs[train_indices], targets[train_indices]
    val_inputs, val_targets = inputs[val_indices], targets[val_indices]
    
    # Create datasets and data loaders
    train_dataset = SDFDataset(train_inputs, train_targets)
    val_dataset = SDFDataset(val_inputs, val_targets)
    
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    input_size = 4  # (x, y, z, shape type)
    hidden_size = 64
    output_size = 1  # SDF value
    num_layers = 5
    
    model = FCNN(input_size, hidden_size, output_size, num_layers).to(device)
    print(f"Model architecture:\n{model}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("\nTraining the model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
    
    # Visualize training process
    visualize_losses(train_losses, val_losses)
    
    # Visualize predictions for each shape type
    for shape_type in range(4):
        visualize_predictions(model, shape_type)
    
    # Save the model to cache
    model_cache_name = 'sdf_4d_model'
    cache_save(model.state_dict(), model_cache_name)
    print(f"Model saved to cache as '{model_cache_name}'")
    
    # Also save the model architecture parameters for loading later
    model_params = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'num_layers': num_layers
    }
    cache_save(model_params, f'{model_cache_name}_params')
    print(f"Model parameters saved to cache as '{model_cache_name}_params'")
    
    return model

if __name__ == "__main__":
    trained_model = main()