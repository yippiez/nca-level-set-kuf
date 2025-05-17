"""
Unified test file for all SDF-related rendering and visualization functionality.

This file combines tests from:
- test_render.py
- test_level_set_render.py
- test_4d_sdf_viz.py
"""

import os
import sys
import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tree.fcnn import FCNN
from util.types import CSGRenderConfig
from util.sdf import (
    sdf_sphere, sdf_box, sdf_pill, sdf_torus, sdf_cylinder,
    sdf_cone, sdf_octahedron, sdf_pyramid, sdf_hexagonal_prism,
    sdf_ellipsoid, sdf_rounded_box, sdf_link, sdf_star,
    sdf_render_csg, sdf_render_csg_animation, 
    sdf_render_level_set, sdf_render_level_set_grid
)


# ==========================================
# Common fixtures used by multiple test classes
# ==========================================

@pytest.fixture
def device():
    """Get the compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ==========================================
# Basic SDF shape tests
# ==========================================

class TestSDFBasicShapes:
    """Test basic SDF shape functions."""
    
    def test_sdf_sphere(self):
        """Test sphere SDF function."""
        # Interior point
        center = np.array([0, 0, 0])
        radius = 1.0
        interior_point = np.array([0.5, 0, 0])
        assert sdf_sphere(interior_point, center, radius) < 0
        
        # Exterior point
        exterior_point = np.array([2, 0, 0])
        assert sdf_sphere(exterior_point, center, radius) > 0
        
        # Surface point (approximately)
        surface_point = np.array([1.001, 0, 0])
        assert abs(sdf_sphere(surface_point, center, radius)) < 0.01
        
        # Batch processing
        points = np.array([
            [0.5, 0, 0],  # interior
            [2.0, 0, 0],  # exterior
            [1.0, 0, 0]   # surface
        ])
        distances = sdf_sphere(points, center, radius)
        assert distances.shape == (3,)
        assert distances[0] < 0  # interior
        assert distances[1] > 0  # exterior
        assert abs(distances[2]) < 0.01  # surface
    
    def test_sdf_box(self):
        """Test box SDF function."""
        center = np.array([0, 0, 0])
        dims = np.array([2.0, 2.0, 2.0])
        
        # Interior point
        interior_point = np.array([0.5, 0.5, 0.5])
        assert sdf_box(interior_point, center, dims) < 0
        
        # Exterior point
        exterior_point = np.array([2, 2, 2])
        assert sdf_box(exterior_point, center, dims) > 0
        
        # Batch processing
        points = np.array([
            [0.5, 0.5, 0.5],  # interior
            [2.0, 2.0, 2.0],  # exterior
            [1.0, 0.0, 0.0]   # near surface
        ])
        distances = sdf_box(points, center, dims)
        assert distances.shape == (3,)
        assert distances[0] < 0  # interior
        assert distances[1] > 0  # exterior


# ==========================================
# Basic CSG Rendering tests
# ==========================================

class TestSDFRendering:
    """Test SDF rendering functions."""
    
    def test_sdf_render_csg(self, temp_dir):
        """Test basic CSG rendering."""
        # Setup simple sphere SDF
        def simple_sphere_sdf(points):
            center = np.array([0, 0, 0])
            radius = 0.5
            return sdf_sphere(points, center, radius)
        
        # Create render config
        config = CSGRenderConfig(
            grid_size=30,  # Smaller grid for faster tests
            bounds=(-1.0, 1.0),
            image_size=(400, 400),
            camera_rotation=(45, 20, 0),
            save_path=os.path.join(temp_dir, "sphere.png")
        )
        
        # Render and verify output file exists
        sdf_render_csg(simple_sphere_sdf, config)
        assert os.path.exists(config.save_path)
        assert os.path.getsize(config.save_path) > 0
    
    def test_sdf_render_csg_animation(self, temp_dir):
        """Test CSG animation rendering."""
        # Setup simple sphere SDF
        def simple_sphere_sdf(points):
            center = np.array([0, 0, 0])
            radius = 0.5
            return sdf_sphere(points, center, radius)
        
        # Create render config (with minimal frames for faster test)
        config = CSGRenderConfig(
            grid_size=20,  # Smaller grid for faster tests
            n_frames=2,    # Minimal animation
            fps=10,
            image_size=(300, 300),
            camera_rotation=(45, 20, 0),
            save_path=os.path.join(temp_dir, "sphere_anim.gif")
        )
        
        # Render and verify output file exists
        sdf_render_csg_animation(simple_sphere_sdf, config)
        assert os.path.exists(config.save_path)
        assert os.path.getsize(config.save_path) > 0
    
    def test_compound_shape(self, temp_dir):
        """Test rendering of compound shapes using CSG operations."""
        def compound_sdf(points):
            # Union of a sphere and a box
            sphere_center = np.array([0.5, 0, 0])
            sphere_radius = 0.4
            sphere_dist = sdf_sphere(points, sphere_center, sphere_radius)
            
            box_center = np.array([-0.5, 0, 0])
            box_dims = np.array([0.6, 0.6, 0.6])
            box_dist = sdf_box(points, box_center, box_dims)
            
            # Union operation (minimum of distances)
            return np.minimum(sphere_dist, box_dist)
        
        # Create render config
        config = CSGRenderConfig(
            grid_size=30,
            bounds=(-2.0, 2.0),
            image_size=(400, 400),
            save_path=os.path.join(temp_dir, "compound.png")
        )
        
        # Render and verify output file exists
        sdf_render_csg(compound_sdf, config)
        assert os.path.exists(config.save_path)
        assert os.path.getsize(config.save_path) > 0


# ==========================================
# Level Set Rendering tests
# ==========================================

class TestLevelSetRendering:
    """Test level set rendering functions."""
    
    def simple_4d_sdf(self, points):
        """Simple 4D SDF for testing."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        shape = points[:, 3]
        
        # Sphere distance
        sphere_dist = np.sqrt(x**2 + y**2 + z**2) - 0.5
        
        # Box distance
        box_dist = np.maximum(np.maximum(np.abs(x) - 0.5, np.abs(y) - 0.5), np.abs(z) - 0.5)
        
        # Morph between them
        return (1 - shape) * sphere_dist + shape * box_dist
    
    def test_level_set_function(self):
        """Test 4D level set SDF function."""
        # Create simple 4D SDF that morphs between sphere and box
        def morphing_sdf(points_4d):
            points_3d = points_4d[..., :3]
            shape_param = points_4d[..., 3]
            
            # If shape_param is a scalar or a 0-dim tensor, reshape to be compatible with points
            if np.isscalar(shape_param) or (hasattr(shape_param, 'ndim') and shape_param.ndim == 0):
                shape_param = np.full(points_3d.shape[0], shape_param)
            
            # Sphere SDF when shape_param = 0
            center = np.array([0, 0, 0])
            sphere_dist = sdf_sphere(points_3d, center, 0.5)
            
            # Box SDF when shape_param = 1
            box_dims = np.array([0.8, 0.8, 0.8])
            box_dist = sdf_box(points_3d, center, box_dims)
            
            # Linear interpolation between sphere and box
            return sphere_dist * (1 - shape_param) + box_dist * shape_param
        
        # Test single point, single shape value
        test_point = np.array([[0.2, 0.2, 0.2, 0.5]])  # x,y,z,shape (as batch)
        result = morphing_sdf(test_point)
        assert isinstance(result, np.ndarray)  # Result should be an array
        
        # Test multiple points, single shape value
        test_points = np.array([
            [0.2, 0.2, 0.2, 0.5],
            [0.7, 0.0, 0.0, 0.5]
        ])
        result = morphing_sdf(test_points)
        assert result.shape == (2,)
        
        # Test multiple shape values
        test_points = np.array([
            [0.2, 0.2, 0.2, 0.0],  # pure sphere
            [0.2, 0.2, 0.2, 1.0]   # pure box
        ])
        result = morphing_sdf(test_points)
        assert result.shape == (2,)
        # Sphere and box should give different distances for the same point
        assert result[0] != result[1]
    
    def test_level_set_animation(self, temp_dir):
        """Test level set animation creation."""
        config = CSGRenderConfig(
            grid_size=30,  # Low resolution for fast test
            bounds=(-1.0, 1.0),
            save_path=os.path.join(temp_dir, "level_set_anim.gif"),
            image_size=(200, 200),
            n_frames=5,
            fps=5
        )
        
        # Test with default shape values
        result_path = sdf_render_level_set(self.simple_4d_sdf, config)
        assert Path(result_path).exists()
        assert result_path.endswith('.gif')
        
        # Test with custom shape values
        shape_values = [0.0, 0.5, 1.0]
        result_path2 = sdf_render_level_set(self.simple_4d_sdf, config, shape_values)
        assert Path(result_path2).exists()
    
    def test_level_set_grid(self, temp_dir):
        """Test level set grid creation."""
        config = CSGRenderConfig(
            grid_size=30,
            bounds=(-1.0, 1.0),
            save_path=os.path.join(temp_dir, "level_set_grid.png"),
            image_size=(150, 150)
        )
        
        # Test with default shape values
        result_path = sdf_render_level_set_grid(self.simple_4d_sdf, config)
        assert Path(result_path).exists()
        assert result_path.endswith('.png')
        
        # Test with custom shape values
        shape_values = [0.0, 0.33, 0.67, 1.0]
        result_path2 = sdf_render_level_set_grid(self.simple_4d_sdf, config, shape_values)
        assert Path(result_path2).exists()
    
    def test_level_set_with_invalid_sdf(self, temp_dir):
        """Test error handling with invalid SDF function."""
        def invalid_sdf(points):
            # Wrong shape - doesn't expect 4D input
            return np.zeros(points.shape[0])
        
        config = CSGRenderConfig(
            grid_size=20,
            bounds=(-1.0, 1.0),
            save_path=os.path.join(temp_dir, "invalid.gif"),
            image_size=(100, 100),
            n_frames=3
        )
        
        # Should raise an error
        with pytest.raises(Exception):
            sdf_render_level_set(invalid_sdf, config)
    
    def test_sdf_render_level_set(self, temp_dir):
        """Test level set rendering."""
        # Create simple 4D SDF that morphs between sphere and box
        def morphing_sdf(points_4d):
            points_3d = points_4d[..., :3]
            shape_param = points_4d[..., 3]
            
            # If shape_param is a scalar or a 0-dim tensor, reshape
            if np.isscalar(shape_param) or (hasattr(shape_param, 'ndim') and shape_param.ndim == 0):
                shape_param = np.full(points_3d.shape[0], shape_param)
            
            # Sphere SDF when shape_param = 0
            center = np.array([0, 0, 0])
            sphere_dist = sdf_sphere(points_3d, center, 0.5)
            
            # Box SDF when shape_param = 1
            box_dims = np.array([0.8, 0.8, 0.8])
            box_dist = sdf_box(points_3d, center, box_dims)
            
            # Linear interpolation between sphere and box
            return sphere_dist * (1 - shape_param) + box_dist * shape_param
        
        # Create render config
        config = CSGRenderConfig(
            grid_size=20,  # Smaller grid for faster tests
            n_frames=2,    # Minimal animation
            fps=10,
            image_size=(300, 300),
            save_path=os.path.join(temp_dir, "level_set.gif")
        )
        
        # Shape values for morphing (0=sphere, 1=box)
        shape_values = np.linspace(0, 1, config.n_frames)
        
        # Render and verify output file exists
        sdf_render_level_set(morphing_sdf, config, shape_values=shape_values)
        assert os.path.exists(config.save_path)
        assert os.path.getsize(config.save_path) > 0
    
    def test_sdf_render_level_set_grid(self, temp_dir):
        """Test level set grid rendering."""
        # Create simple 4D SDF that morphs between sphere and box
        def morphing_sdf(points_4d):
            points_3d = points_4d[..., :3]
            shape_param = points_4d[..., 3]
            
            # If shape_param is a scalar or a 0-dim tensor, reshape
            if np.isscalar(shape_param) or (hasattr(shape_param, 'ndim') and shape_param.ndim == 0):
                shape_param = np.full(points_3d.shape[0], shape_param)
            
            # Sphere SDF when shape_param = 0
            center = np.array([0, 0, 0])
            sphere_dist = sdf_sphere(points_3d, center, 0.5)
            
            # Box SDF when shape_param = 1
            box_dims = np.array([0.8, 0.8, 0.8])
            box_dist = sdf_box(points_3d, center, box_dims)
            
            # Linear interpolation between sphere and box
            return sphere_dist * (1 - shape_param) + box_dist * shape_param
        
        # Create render config
        config = CSGRenderConfig(
            grid_size=20,  # Smaller grid for faster tests
            image_size=(300, 300),
            save_path=os.path.join(temp_dir, "level_set_grid.png")
        )
        
        # Shape values for grid (0=sphere, 1=box)
        shape_values = np.linspace(0, 1, 4)  # 4 shapes for faster test
        
        # Render and verify output file exists
        sdf_render_level_set_grid(morphing_sdf, config, shape_values=shape_values)
        assert os.path.exists(config.save_path)
        assert os.path.getsize(config.save_path) > 0


# ==========================================
# Neural Network SDF tests
# ==========================================

@pytest.fixture
def test_model(device):
    """Create a test 4D SDF model for testing."""
    # Create a small test model
    input_size = 4   # x, y, z, shape
    hidden_size = 32  # Smaller for testing
    output_size = 1   # SDF value
    num_layers = 3    # Smaller for testing
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    model = FCNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers
    ).to(device)
    
    model.eval()
    return model


class TestNeuralNetworkSDF:
    """Test neural network based SDF rendering."""
    
    @pytest.fixture
    def simple_fcnn_model(self):
        """Create a simple FCNN model for testing."""
        input_size = 4  # x, y, z, shape
        hidden_size = 32  # Small for testing
        output_size = 1  # SDF value
        num_layers = 3   # Small for testing
        
        # Initialize with deterministic weights for testing
        torch.manual_seed(42)
        model = FCNN(input_size, hidden_size, output_size, num_layers)
        
        return model

    def test_model_predictions(self, test_model, device):
        """Test model predictions for specific points and shape values."""
        test_points = np.array([
            [0, 0, 0],      # Center
            [0.5, 0, 0],    # Right
            [0, 0.5, 0],    # Up
            [0, 0, 0.5],    # Forward
            [0.3, 0.3, 0],  # Diagonal in XY plane
        ])
        
        shape_values = [0, 0.5, 1, 2, 3]
        results = {}
        
        with torch.no_grad():
            for shape in shape_values:
                # Prepare inputs
                shape_indices = np.ones((len(test_points), 1)) * shape
                inputs = np.hstack([test_points, shape_indices])
                inputs_tensor = torch.FloatTensor(inputs).to(device)
                
                # Get predictions
                predictions = test_model(inputs_tensor).cpu().numpy()
                results[shape] = predictions
        
        # Basic sanity checks
        assert len(results) == len(shape_values)
        for shape in shape_values:
            assert results[shape].shape == (5, 1)
    
    def test_model_interpolation(self, test_model, device):
        """Test that model can interpolate between shapes."""
        # Test point at origin
        test_point = np.array([[0, 0, 0]])
        
        # Test interpolation from shape 0 to shape 1
        interpolation_values = np.linspace(0, 1, 5)  # Reduced count for speed
        distances = []
        
        with torch.no_grad():
            for s in interpolation_values:
                shape_indices = np.ones((1, 1)) * s
                inputs = np.hstack([test_point, shape_indices])
                inputs_tensor = torch.FloatTensor(inputs).to(device)
                
                prediction = test_model(inputs_tensor).cpu().numpy()[0, 0]
                distances.append(prediction)
        
        # Check that we got results for all interpolation values
        assert len(distances) == len(interpolation_values)
    
    def test_fcnn_sdf_function(self, simple_fcnn_model):
        """Test FCNN-based SDF function."""
        device = torch.device('cpu')
        model = simple_fcnn_model.to(device)
        model.eval()
        
        # Create an SDF function wrapper
        def fcnn_sdf(points_4d, shape_param=None):
            # Convert to tensor
            if not isinstance(points_4d, torch.Tensor):
                points_4d = torch.FloatTensor(points_4d)
            
            original_shape = points_4d.shape
            
            # Handle 3D points with separate shape parameter
            if len(original_shape) == 2 and original_shape[1] == 3 and shape_param is not None:
                batch_size = points_4d.shape[0]
                shape_param_tensor = torch.full((batch_size, 1), shape_param, dtype=torch.float32)
                points_4d = torch.cat([points_4d, shape_param_tensor], dim=1)
            
            # Ensure 4D points
            assert points_4d.shape[-1] == 4, "Points must be 4D (x,y,z,shape)"
            
            # Reshape for batch processing if needed
            if len(original_shape) > 2:
                points_4d = points_4d.reshape(-1, 4)
            
            # Move to device and get predictions
            points_4d = points_4d.to(device)
            with torch.no_grad():
                sdf_values = model(points_4d).cpu().numpy()
            
            # Reshape back if needed
            if len(original_shape) > 2:
                sdf_values = sdf_values.reshape(original_shape[:-1])
            else:
                sdf_values = sdf_values.reshape(-1)
            
            if sdf_values.size == 1:
                return float(sdf_values.item())
            return sdf_values
        
        # Test single point
        test_point = np.array([0.2, 0.2, 0.2, 0.5])
        result = fcnn_sdf(test_point)
        assert isinstance(result, float)
        
        # Test multiple points
        test_points = np.array([
            [0.2, 0.2, 0.2, 0.0],
            [0.2, 0.2, 0.2, 1.0]
        ])
        result = fcnn_sdf(test_points)
        assert result.shape == (2,)
        
        # Test 3D points with shape parameter
        test_points = np.array([
            [0.2, 0.2, 0.2],
            [0.7, 0.0, 0.0]
        ])
        result = fcnn_sdf(test_points, shape_param=0.5)
        assert result.shape == (2,)
    
    def test_fcnn_render_level_set(self, simple_fcnn_model, temp_dir):
        """Test rendering level sets with FCNN model."""
        device = torch.device('cpu')
        model = simple_fcnn_model.to(device)
        model.eval()
        
        # Create an SDF function wrapper
        def fcnn_sdf(points_4d, shape_param=None):
            # Convert to tensor
            if not isinstance(points_4d, torch.Tensor):
                points_4d = torch.FloatTensor(points_4d)
            
            original_shape = points_4d.shape
            
            # Handle 3D points with separate shape parameter
            if len(original_shape) == 2 and original_shape[1] == 3 and shape_param is not None:
                batch_size = points_4d.shape[0]
                shape_param_tensor = torch.full((batch_size, 1), shape_param, dtype=torch.float32)
                points_4d = torch.cat([points_4d, shape_param_tensor], dim=1)
            
            # Ensure 4D points
            assert points_4d.shape[-1] == 4, "Points must be 4D (x,y,z,shape)"
            
            # Reshape for batch processing if needed
            if len(original_shape) > 2:
                points_4d = points_4d.reshape(-1, 4)
            
            # Move to device and get predictions
            points_4d = points_4d.to(device)
            with torch.no_grad():
                sdf_values = model(points_4d).cpu().numpy()
            
            # Reshape back if needed
            if len(original_shape) > 2:
                sdf_values = sdf_values.reshape(original_shape[:-1])
            else:
                sdf_values = sdf_values.reshape(-1)
            
            if sdf_values.size == 1:
                return float(sdf_values.item())
            return sdf_values
        
        # Create render config
        config = CSGRenderConfig(
            grid_size=20,  # Smaller grid for faster tests
            n_frames=2,    # Minimal animation
            fps=10,
            image_size=(300, 300),
            save_path=os.path.join(temp_dir, "fcnn_level_set.gif")
        )
        
        # Shape values for morphing
        shape_values = np.linspace(0, 1, config.n_frames)
        
        # Add error handling for marching cubes
        def safe_fcnn_sdf(points_4d):
            values = fcnn_sdf(points_4d)
            # Clip values to prevent marching cubes errors
            if isinstance(values, np.ndarray):
                return np.clip(values, -1.5, 1.5)
            return max(-1.5, min(1.5, values))
        
        # Render and verify output file exists
        sdf_render_level_set(safe_fcnn_sdf, config, shape_values=shape_values)
        assert os.path.exists(config.save_path)
        assert os.path.getsize(config.save_path) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])