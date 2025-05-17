"""FCNN-based experiment implementation."""

import os
from pathlib import Path
import time
import json
from typing import Any, Callable, Union, ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pydantic import BaseModel, Field

from stok.tree.point_based import PointBasedExperiment, PointSampleStrategy, PointBasedExperimentResult
from stok.util.sdf.similarity import sdf_get_sampled_boolean_similarity
from stok.util.types import LayerDetails


def fcnn_n_perceptrons(model: nn.Module) -> int:
    """Count the total number of perceptrons in a FCNN model."""
    total_perceptrons = 0
    
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_perceptrons += module.out_features
    
    return total_perceptrons


def fcnn_layer_details(model: nn.Module) -> list[LayerDetails]:
    """Get detailed information about each layer in a FCNN model."""
    layer_details = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_details.append(LayerDetails(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                n_neurons=module.out_features
            ))
    
    return layer_details



# Pydantic models for parameters and results
class FCNNModelParams(BaseModel):
    """Parameters for configuring the FCNN model."""
    input_size: int = Field(default=3, description="Number of input features (3 for xyz or 4 for xyz+shape)")
    hidden_size: int = Field(default=64, description="Size of hidden layers")
    output_size: int = Field(default=1, description="Number of output features (typically 1 for SDF)")
    num_layers: int = Field(default=5, description="Total number of layers including input and output")

class FCNNTrainParams(BaseModel):
    """Parameters for training the FCNN model."""
    batch_size: int = Field(default=64, description="Batch size for training")
    learning_rate: float = Field(default=0.001, description="Learning rate for optimizer")
    num_epochs: int = Field(default=1000, description="Maximum number of training epochs")
    # Removed val_split and patience as we want to overfit

class FCNNExperimentMetrics(BaseModel):
    """Metrics from training and evaluating the FCNN model."""
    boolean_similarity: float = Field(description="Boolean similarity between the network output and the ground truth SDF")
    training_time: float = Field(description="Total training time in seconds")
    epochs: int = Field(description="Number of epochs the model was trained")
    loss: float = Field(description="Final training loss value")

class FCNNExperimentResult(BaseModel):
    """Results from the FCNN experiment."""
    data_size: int = Field(description="Number of data points used")
    model_path: str = Field(description="Path where the model was saved")
    dump_path: str = Field(description="Path where all experiment outputs will be saved")
    metrics: FCNNExperimentMetrics = Field(description="Performance metrics")
    model_params: FCNNModelParams = Field(description="Model parameters used")
    train_params: FCNNTrainParams = Field(description="Training parameters used")
    
# FCNN model definition (moved from models/fcnn.py)
class FCNN(nn.Module):
    """Fully Connected Neural Network model for learning SDFs."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int) -> None:
        """Initialize the FCNN model.
        
        Args:
            input_size: Number of input features (3 for xyz or 4 for xyz+shape)
            hidden_size: Size of hidden layers
            output_size: Number of output features (typically 1 for SDF)
            num_layers: Total number of layers including input and output
        """
        super(FCNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


class FCNNExperiment(PointBasedExperiment):
    """FCNN-based experiment for learning SDFs.
    
    This class implements a fully connected neural network approach
    for learning signed distance functions from sampled points.
    """
    
    NAME: ClassVar[str] = "FCNNExperiment"
    VERSION: ClassVar[str] = "v1a1"
    
    def __init__(self, 
                 sample_strategy: PointSampleStrategy,
                 sdf: Callable[[np.ndarray], np.ndarray],
                 bound_begin: Union[np.ndarray, tuple[float, float, float]],
                 bound_end: Union[np.ndarray, tuple[float, float, float]],
                 model_params: Union[FCNNModelParams, dict[str, Any]],
                 train_params: Union[FCNNTrainParams, dict[str, Any]],
                 model_name: str) -> None:
        """Initialize the FCNN experiment.
        
        Args:
            sample_strategy: Strategy for sampling points
            sdf: SDF function to evaluate
            bound_begin: Lower bounds for sampling points
            bound_end: Upper bounds for sampling points
            model_params: Parameters for the FCNN model
                        (input_size, hidden_size, output_size, num_layers)
            train_params: Training parameters
                        (batch_size, learning_rate, num_epochs)
            model_name: Name of the model (used for naming model files in reports)
        """
        super().__init__(sample_strategy, sdf, bound_begin, bound_end)
        self.model_params = model_params
        self.train_params = train_params
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = None

    @property
    def dump_path(self) -> Path:
        """Get the path to dump the experiment results."""
        original_dump_path = super().dump_path

        experiment_dir = original_dump_path / self.model_name
        os.makedirs(experiment_dir, exist_ok=True)

        return experiment_dir
    
    def do(self) -> FCNNExperimentResult:
        """Run the FCNN experiment.
        
        This method performs the following steps:
        1. Sample points from SDF
        2. Prepare data for training
        3. Create and train the FCNN model to overfit
        4. Evaluate the model using boolean similarity
        5. Save the model and results
        
        Returns:
            FCNNExperimentResult: Structured results from the experiment
        """
        print(f"Running {self.NAME} ({self.VERSION}) on {self.device}")
        
        # Sample points and prepare data
        base_results: PointBasedExperimentResult = super().do()
        
        X = base_results.points
        y = base_results.distances.reshape(-1, 1) 
        
        # Convert to PyTorch tensors for the full dataset
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create DataLoader with the full dataset (no validation split)
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(
            dataset,
            batch_size=self.train_params.batch_size,
            shuffle=True
        )
        
        # Create the model
        self.model = FCNN(
            input_size=self.model_params.input_size,
            hidden_size=self.model_params.hidden_size,
            output_size=self.model_params.output_size,
            num_layers=self.model_params.num_layers
        ).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_params.learning_rate
        )
        
        # Training loop
        start_time = time.time()
        history = {
            'train_loss': []
        }
        
        num_epochs = self.train_params.num_epochs
        final_loss = 0.0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in data_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            
            # Calculate average loss for this epoch
            epoch_loss /= len(data_loader.dataset)
            history['train_loss'].append(epoch_loss)
            final_loss = epoch_loss
            
            # Print progress every N epochs
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: loss={epoch_loss:.6f}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Save model
        model_path = os.path.join(self.dump_path, f'{self.model_name}.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # Store history for later use
        self.history = history
        
        # Save training history
        history_path = os.path.join(self.dump_path, f'{self.model_name}_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Evaluate the model using boolean similarity
        # Create a prediction function from our trained model
        def model_sdf(points: np.ndarray) -> np.ndarray:
            """Convert model into an SDF function compatible with sdf_get_sampled_boolean_similarity."""
            with torch.no_grad():
                points_tensor = torch.FloatTensor(points).to(self.device)
                predictions = self.model(points_tensor).cpu().numpy()
            return predictions.flatten()
        
        # Use the original SDF function
        def original_sdf(points: np.ndarray) -> np.ndarray:
            """Wrapper for the original SDF function."""
            return self.sdf(points).flatten()
        
        # Calculate boolean similarity
        bound_begin = tuple(self.bound_begin) if isinstance(self.bound_begin, np.ndarray) else self.bound_begin
        bound_end = tuple(self.bound_end) if isinstance(self.bound_end, np.ndarray) else self.bound_end
        step_size = 0.1  # Adjust this based on desired precision vs. computation time
        
        boolean_similarity = sdf_get_sampled_boolean_similarity(
            model_sdf,
            original_sdf,
            bound_begin,
            bound_end,
            step_size
        )
        
        print(f"Boolean similarity: {boolean_similarity:.4f}")
        
        # Create metrics
        # TODO add n_perceptron as metric here
        metrics = FCNNExperimentMetrics(
            boolean_similarity=float(boolean_similarity),
            training_time=float(total_time),
            epochs=num_epochs,
            loss=float(final_loss)
        )
        
        # Create result object
        dump_path = os.path.join(self.dump_path, f'{self.model_name}_experiment_result.json')
        result = FCNNExperimentResult(
            data_size=len(X),
            model_path=model_path,
            dump_path=self.dump_path,
            metrics=metrics,
            model_params=self.model_params,
            train_params=self.train_params
        )
        
        # Save the result as JSON
        with open(dump_path, 'w') as f:
            f.write(json.dumps(result.model_dump(), indent=2))
                
        return result
    

    def show_as_side_by_side(self):
        raise NotImplementedError("This method is not implemented yet. ")

    def show_as_animation(self):
        pass

    def show_as_grid(self):
        pass

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            points: Input points (Nx3)
            
        Returns:
            np.ndarray: Predicted SDF values
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")
        
        self.model.eval()
        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(self.device)
            predictions = self.model(points_tensor).cpu().numpy()
        
        return predictions
