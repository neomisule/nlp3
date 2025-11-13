#This module provides training utilities including optimizer factory, training engine, and gradient clipping functionality.

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

class OptimizerFactory:
    #Supports Adam, SGD, and RMSProp optimizers with customizable learning rates and optimizer-specific parameters.
    
    @staticmethod
    def create_optimizer(
        model_parameters,
        optimizer_type: str,
        learning_rate: float = 0.001,
        **kwargs
    ) -> optim.Optimizer:
        """
        Create an optimizer instance based on the specified type and parameters.
        
        Args:
            model_parameters: Model parameters to optimize
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate for the optimizer (default: 0.001)
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            torch.optim.Optimizer: Configured optimizer instance
            
        Raises:
            ValueError: If optimizer_type is not supported
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(
                model_parameters,
                lr=learning_rate,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0)
            )
        
        elif optimizer_type == 'sgd':
            return optim.SGD(
                model_parameters,
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0),
                nesterov=kwargs.get('nesterov', False)
            )
        
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                model_parameters,
                lr=learning_rate,
                alpha=kwargs.get('alpha', 0.99),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0),
                momentum=kwargs.get('momentum', 0)
            )
    
    @staticmethod
    def get_default_parameters(optimizer_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific optimizer type.
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            
        Returns:
            Dict[str, Any]: Default parameters for the optimizer
        """
        optimizer_type = optimizer_type.lower()
        
        defaults = {
            'adam': {
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0
            },
            'sgd': {
                'momentum': 0.9,
                'weight_decay': 0,
                'nesterov': False
            },
            'rmsprop': {
                'alpha': 0.99,
                'eps': 1e-8,
                'weight_decay': 0,
                'momentum': 0
            }
        }
        
        return defaults.get(optimizer_type, {})

class GradientClipper:
    """
    Utility class for handling gradient clipping during training.
    
    Provides functionality to clip gradients by norm and track gradient
    statistics for stability analysis.
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum norm for gradient clipping (default: 1.0)
            norm_type: Type of norm to use for clipping (default: 2.0)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.gradient_norms = []
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients of model parameters and return the gradient norm.
        
        Args:
            model: PyTorch model whose gradients to clip
            
        Returns:
            float: Gradient norm before clipping
        """
        # Calculate gradient norm before clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm, 
            norm_type=self.norm_type
        )
        
        # Store gradient norm for analysis
        self.gradient_norms.append(total_norm.item())
        
        return total_norm.item()
    
    def get_gradient_statistics(self) -> Dict[str, float]:
        """
        Get statistics about gradient norms during training.
        
        Returns:
            Dict[str, float]: Statistics including mean, max, min gradient norms
        """
        if not self.gradient_norms:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
        
        import numpy as np
        norms = np.array(self.gradient_norms)
        
        return {
            'mean': float(np.mean(norms)),
            'max': float(np.max(norms)),
            'min': float(np.min(norms)),
            'std': float(np.std(norms))
        }
    
    def reset_statistics(self):
        """Reset gradient norm statistics."""
        self.gradient_norms = []


@dataclass
class TrainingMetrics:
    """Container for training metrics from a single epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_time: float
    gradient_norm: Optional[float] = None


class Trainer:
    """
    Main training engine for RNN models.
    
    Handles the complete training process including epoch management,
    loss tracking, validation, and timing measurements.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_type: str,
        learning_rate: float = 0.001,
        gradient_clipping: bool = False,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device('cpu'),
        **optimizer_kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate for optimizer (default: 0.001)
            gradient_clipping: Whether to apply gradient clipping (default: False)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            device: Device to use for training 
            **optimizer_kwargs: Additional optimizer parameters
        """
        self.model = model.to(device)
        self.device = device
        self.gradient_clipping = gradient_clipping
        
        # Create optimizer
        self.optimizer = OptimizerFactory.create_optimizer(
            self.model.parameters(),
            optimizer_type,
            learning_rate,
            **optimizer_kwargs
        )
        
        # Initialize gradient clipper if needed
        self.gradient_clipper = None
        if gradient_clipping:
            self.gradient_clipper = GradientClipper(max_norm=max_grad_norm)
        
        # Loss function for binary classification
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = []
        self.loss_history = []
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None
    ) -> TrainingMetrics:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            
        Returns:
            TrainingMetrics: Metrics from the training epoch
        """
        epoch_start_time = time.time()
        
        # Training phase
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        gradient_norm = None
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            targets = targets.float().unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if enabled
            if self.gradient_clipping and self.gradient_clipper:
                gradient_norm = self.gradient_clipper.clip_gradients(self.model)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        val_loss, val_accuracy = 0.0, 0.0
        if val_loader is not None:
            val_loss, val_accuracy = self.validate(val_loader)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Create metrics object
        metrics = TrainingMetrics(
            epoch=len(self.training_history) + 1,
            train_loss=avg_train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            epoch_time=epoch_time,
            gradient_norm=gradient_norm
        )
        
        # Store metrics
        self.training_history.append(metrics)
        self.loss_history.append(avg_train_loss)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple[float, float]: Validation loss and accuracy
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                targets = targets.float().unsqueeze(1)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        return avg_val_loss, val_accuracy
    
    def train_multiple_epochs(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> List[TrainingMetrics]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            val_loader: Optional DataLoader for validation data
            verbose: Whether to print training progress
            
        Returns:
            List[TrainingMetrics]: List of metrics for each epoch
        """
        epoch_metrics = []
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            epoch_metrics.append(metrics)
            
            if verbose:
                print(f"Epoch {metrics.epoch}/{num_epochs}: "
                      f"Train Loss: {metrics.train_loss:.4f}, "
                      f"Train Acc: {metrics.train_accuracy:.4f}, "
                      f"Val Loss: {metrics.val_loss:.4f}, "
                      f"Val Acc: {metrics.val_accuracy:.4f}, "
                      f"Time: {metrics.epoch_time:.2f}s")
                
                if metrics.gradient_norm is not None:
                    print(f"  Gradient Norm: {metrics.gradient_norm:.4f}")
        
        return epoch_metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training process.
        
        Returns:
            Dict[str, Any]: Training summary including timing and gradient statistics
        """
        if not self.training_history:
            return {}
        
        total_time = sum(m.epoch_time for m in self.training_history)
        avg_epoch_time = total_time / len(self.training_history)
        
        summary = {
            'total_epochs': len(self.training_history),
            'total_training_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'final_train_loss': self.training_history[-1].train_loss,
            'final_train_accuracy': self.training_history[-1].train_accuracy,
            'loss_history': self.loss_history
        }
        
        # Add gradient statistics if gradient clipping was used
        if self.gradient_clipper:
            summary['gradient_statistics'] = self.gradient_clipper.get_gradient_statistics()
        
        return summary