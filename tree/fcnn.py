"""FCNN-based experiment implementation."""

import os
import time
import json
from typing import Dict, Any, Callable, Tuple, Union, Optional, ClassVar, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pydantic import BaseModel, Field

from .point_based import PointBasedExperiment, PointSampleStrategy
from util.paths import get_reports_dir
from util.sdf.similarity import sdf_get_sampled_boolean_similarity


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
    report_path: str = Field(description="Path where the experiment report was saved as JSON")
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
                 bound_begin: Union[np.ndarray, Tuple[float, float, float]],
                 bound_end: Union[np.ndarray, Tuple[float, float, float]],
                 model_params: Optional[Union[FCNNModelParams, Dict[str, Any]]] = None,
                 train_params: Optional[Union[FCNNTrainParams, Dict[str, Any]]] = None,
                 experiment_name: str = "fcnn_experiment") -> None:
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
            experiment_name: Name of the experiment (used for saving results)
        """
        super().__init__(sample_strategy, sdf, bound_begin, bound_end)
        
        # Convert dict to Pydantic model if needed
        if model_params is None:
            self.model_params = FCNNModelParams()
        elif isinstance(model_params, dict):
            self.model_params = FCNNModelParams(**model_params)
        else:
            self.model_params = model_params
        
        # Convert dict to Pydantic model if needed
        if train_params is None:
            self.train_params = FCNNTrainParams()
        elif isinstance(train_params, dict):
            self.train_params = FCNNTrainParams(**train_params)
        else:
            self.train_params = train_params
        
        self.experiment_name = experiment_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = None
    
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
        base_results = super().do()
        data = base_results['data']
        
        # Prepare input features and target values
        # Assuming data is now a numpy array with columns [x, y, z, distance]
        X = data[:, :3]  # First 3 columns are x, y, z
        y = data[:, 3:4]  # Last column is distance
        
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
        
        # Create reports directory
        reports_dir = get_reports_dir(self.experiment_name)
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(reports_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Store history for later use
        self.history = history
        
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
        metrics = FCNNExperimentMetrics(
            boolean_similarity=float(boolean_similarity),
            training_time=float(total_time),
            epochs=num_epochs,
            loss=float(final_loss)
        )
        
        # Create result object
        report_path = os.path.join(reports_dir, 'experiment_result.json')
        result = FCNNExperimentResult(
            data_size=len(X),
            model_path=model_path,
            report_path=report_path,
            metrics=metrics,
            model_params=self.model_params,
            train_params=self.train_params
        )
        
        # Save the result as JSON
        with open(report_path, 'w') as f:
            # Use model_dump for Pydantic v2
            if hasattr(result, 'model_dump'):
                f.write(json.dumps(result.model_dump(), indent=2))
            # Use dict() for Pydantic v1
            else:
                f.write(json.dumps(result.dict(), indent=2))
                
        return result
    
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
