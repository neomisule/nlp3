"""
Evaluation and metrics system for RNN sentiment classification models.

This module provides comprehensive evaluation capabilities including
accuracy calculation, F1-score computation, and timing measurements.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .config import ExperimentResult, ExperimentConfig


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics from model assessment."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    loss: float
    inference_time: float
    total_samples: int


class ModelEvaluator:
    """
    Comprehensive model evaluator for binary sentiment classification.
    
    Provides accuracy calculation, macro F1-score computation using scikit-learn,
    and timing measurement utilities for training epochs and inference.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize model evaluator.
        
        Args:
            device: Device to use for evaluation (default: automatically detects GPU if available)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define loss function
        self.criterion = nn.BCELoss()
        
        print(f"Using device: {self.device}")
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        verbose: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate model on test data with comprehensive metrics.
        
        Args:
            model: PyTorch model to evaluate
            test_loader: DataLoader for test data
            verbose: Whether to print evaluation progress
            
        Returns:
            EvaluationMetrics: Comprehensive evaluation metrics
        """
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        inference_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                targets_float = targets.float().unsqueeze(1)  # Shape: (batch_size, 1)
                
                # Forward pass
                outputs = model(data)
                loss = self.criterion(outputs, targets_float)
                
                # Convert outputs to binary predictions
                predictions = (outputs > 0.5).float().squeeze()
                
                # Store predictions and targets for metric calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                total_loss += loss.item()
                
                if verbose and batch_idx % 100 == 0:
                    print(f"Evaluated batch {batch_idx}/{len(test_loader)}")
        
        inference_time = time.time() - inference_start_time
        
        # Convert to numpy arrays for sklearn metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        avg_loss = total_loss / len(test_loader)
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            loss=avg_loss,
            inference_time=inference_time,
            total_samples=len(all_targets)
        )
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score (Macro): {f1:.4f}")
            print(f"Precision (Macro): {precision:.4f}")
            print(f"Recall (Macro): {recall:.4f}")
            print(f"Loss: {avg_loss:.4f}")
            print(f"Inference Time: {inference_time:.2f}s")
            print(f"Total Samples: {len(all_targets)}")
        
        return metrics
    
    def calculate_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Calculate binary classification accuracy.
        
        Args:
            predictions: Model predictions (probabilities or logits)
            targets: Ground truth binary labels
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        # Convert predictions to binary (threshold at 0.5)
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        
        binary_preds = (predictions > 0.5).float()
        
        # Ensure targets are float for comparison
        targets = targets.float()
        
        # Calculate accuracy
        correct = (binary_preds == targets).sum().item()
        total = targets.size(0)
        
        return correct / total
    
    def calculate_f1_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        average: str = 'macro'
    ) -> float:
        """
        Calculate F1-score using scikit-learn.
        
        Args:
            predictions: Model predictions (probabilities or logits)
            targets: Ground truth binary labels
            average: Averaging strategy ('macro', 'micro', 'binary')
            
        Returns:
            float: F1-score
        """
        # Convert to numpy arrays
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        
        binary_preds = (predictions > 0.5).float()
        
        y_true = targets.cpu().numpy()
        y_pred = binary_preds.cpu().numpy()
        
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    def measure_inference_time(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        num_runs: int = 1
    ) -> Dict[str, float]:
        """
        Measure model inference timing.
        
        Args:
            model: PyTorch model to time
            test_loader: DataLoader for test data
            num_runs: Number of timing runs for averaging
            
        Returns:
            Dict[str, float]: Timing statistics
        """
        model.eval()
        model.to(self.device)
        
        run_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(self.device)
                    _ = model(data)
            
            run_time = time.time() - start_time
            run_times.append(run_time)
        
        return {
            'mean_time': np.mean(run_times),
            'std_time': np.std(run_times),
            'min_time': np.min(run_times),
            'max_time': np.max(run_times),
            'total_samples': len(test_loader.dataset),
            'avg_time_per_sample': np.mean(run_times) / len(test_loader.dataset)
        }
    
    def evaluate_training_epoch_timing(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None
    ) -> float:
        """
        Measure training time for a single epoch.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            criterion: Loss function (default: BCELoss)
            
        Returns:
            float: Training time in seconds for one epoch
        """
        if criterion is None:
            criterion = self.criterion
        
        model.train()
        model.to(self.device)
        
        epoch_start_time = time.time()
        
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            targets = targets.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - epoch_start_time
        return epoch_time
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of model_name -> model pairs
            test_loader: DataLoader for test data
            verbose: Whether to print comparison results
            
        Returns:
            Dict[str, EvaluationMetrics]: Results for each model
        """
        results = {}
        
        for model_name, model in models.items():
            if verbose:
                print(f"\nEvaluating {model_name}...")
            
            metrics = self.evaluate_model(model, test_loader, verbose=False)
            results[model_name] = metrics
            
            if verbose:
                print(f"{model_name} - Accuracy: {metrics.accuracy:.4f}, "
                      f"F1: {metrics.f1_score:.4f}, Time: {metrics.inference_time:.2f}s")
        
        if verbose:
            # Print comparison summary
            print(f"\n{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10}")
            print("-" * 50)
            for name, metrics in results.items():
                print(f"{name:<20} {metrics.accuracy:<10.4f} "
                      f"{metrics.f1_score:<10.4f} {metrics.inference_time:<10.2f}")
        
        return results


class MetricsCalculator:
    """
    Utility class for metric computation and statistical analysis.
    
    Provides helper functions for calculating various performance metrics
    and statistical measures for model evaluation.
    """
    
    @staticmethod
    def calculate_confusion_matrix_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics from confusion matrix for binary classification.
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Dict[str, float]: Dictionary containing TP, TN, FP, FN, and derived metrics
        """
        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'true_positives': float(tp),
            'true_negatives': float(tn),
            'false_positives': float(fp),
            'false_negatives': float(fn),
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'accuracy': accuracy
        }
    
    @staticmethod
    def calculate_statistical_significance(
        results1: List[float],
        results2: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between two sets of results.
        
        Args:
            results1: First set of results (e.g., accuracy scores)
            results2: Second set of results
            alpha: Significance level (default: 0.05)
            
        Returns:
            Dict[str, Any]: Statistical test results
        """
        from scipy import stats
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(results1, results2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(results1) - 1) * np.var(results1, ddof=1) + 
                             (len(results2) - 1) * np.var(results2, ddof=1)) / 
                            (len(results1) + len(results2) - 2))
        
        cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'cohens_d': cohens_d,
            'mean_diff': np.mean(results1) - np.mean(results2),
            'alpha': alpha
        }
    
    @staticmethod
    def aggregate_metrics(
        metrics_list: List[EvaluationMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics from multiple evaluation runs.
        
        Args:
            metrics_list: List of EvaluationMetrics objects
            
        Returns:
            Dict[str, Dict[str, float]]: Aggregated statistics for each metric
        """
        if not metrics_list:
            return {}
        
        # Extract metric values
        accuracies = [m.accuracy for m in metrics_list]
        f1_scores = [m.f1_score for m in metrics_list]
        precisions = [m.precision for m in metrics_list]
        recalls = [m.recall for m in metrics_list]
        losses = [m.loss for m in metrics_list]
        inference_times = [m.inference_time for m in metrics_list]
        
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return {
            'accuracy': calculate_stats(accuracies),
            'f1_score': calculate_stats(f1_scores),
            'precision': calculate_stats(precisions),
            'recall': calculate_stats(recalls),
            'loss': calculate_stats(losses),
            'inference_time': calculate_stats(inference_times)
        }

import csv
import json
import os
import platform
import psutil
from datetime import datetime
from pathlib import Path


class ResultsLogger:
    """
    Results logging system for experimental results.
    
    Creates CSV output format for experimental results, implements structured
    result storage with all required metrics, and adds metadata logging
    including hardware specs and configuration.
    """
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize results logger.
        
        Args:
            results_dir: Directory to store results (default: 'results')
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of results
        (self.results_dir / 'metrics').mkdir(exist_ok=True)
        (self.results_dir / 'plots').mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        (self.results_dir / 'logs').mkdir(exist_ok=True)
        
        # Hardware and system information
        self.system_info = self._collect_system_info()
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect hardware and system specifications.
        
        Returns:
            Dict[str, Any]: System information dictionary
        """
        try:
            cpu_info = {
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            memory_info = psutil.virtual_memory()._asdict()
            
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'python_version': platform.python_version(),
                'cpu': cpu_info,
                'memory': memory_info,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__
            }
            
            return system_info
            
        except Exception as e:
            print(f"Warning: Could not collect complete system info: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'error': str(e)
            }
    
    def log_experiment_result(
        self,
        result: ExperimentResult,
        experiment_id: str = None
    ) -> str:
        """
        Log a single experiment result to CSV and JSON formats.
        
        Args:
            result: ExperimentResult object containing all metrics
            experiment_id: Optional experiment identifier
            
        Returns:
            str: Path to the logged result file
        """
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare result data
        result_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            
            # Configuration
            'architecture': result.config.architecture,
            'activation': result.config.activation,
            'optimizer': result.config.optimizer,
            'sequence_length': result.config.sequence_length,
            'gradient_clipping': result.config.gradient_clipping,
            'learning_rate': result.config.learning_rate,
            'batch_size': result.config.batch_size,
            'epochs': result.config.epochs,
            'dropout': result.config.dropout,
            
            # Metrics
            'accuracy': result.accuracy,
            'f1_score': result.f1_score,
            'avg_epoch_time': result.avg_epoch_time,
            'total_training_time': result.total_training_time,
            'final_loss': result.final_loss,
            
            # System info
            'platform': self.system_info.get('platform', 'unknown'),
            'cpu_count': self.system_info.get('cpu', {}).get('cpu_count', 'unknown'),
            'memory_total_gb': round(
                self.system_info.get('memory', {}).get('total', 0) / (1024**3), 2
            )
        }
        
        # Save to CSV
        csv_path = self.results_dir / 'metrics' / f'experiment_{experiment_id}.csv'
        self._save_to_csv(result_data, csv_path)
        
        # Save detailed JSON with loss history and system info
        json_data = result_data.copy()
        json_data['loss_history'] = result.loss_history
        json_data['system_info'] = self.system_info
        
        json_path = self.results_dir / 'metrics' / f'experiment_{experiment_id}.json'
        self._save_to_json(json_data, json_path)
        
        return str(csv_path)
    
    def log_multiple_experiments(
        self,
        results: List[ExperimentResult],
        batch_id: str = None
    ) -> str:
        """
        Log multiple experiment results to a single CSV file.
        
        Args:
            results: List of ExperimentResult objects
            batch_id: Optional batch identifier
            
        Returns:
            str: Path to the batch results file
        """
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = self.results_dir / 'metrics' / f'batch_results_{batch_id}.csv'
        
        # Prepare data for all experiments
        all_data = []
        for i, result in enumerate(results):
            result_data = {
                'experiment_id': f"{batch_id}_{i:03d}",
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                
                # Configuration
                'architecture': result.config.architecture,
                'activation': result.config.activation,
                'optimizer': result.config.optimizer,
                'sequence_length': result.config.sequence_length,
                'gradient_clipping': result.config.gradient_clipping,
                'learning_rate': result.config.learning_rate,
                'batch_size': result.config.batch_size,
                'epochs': result.config.epochs,
                'dropout': result.config.dropout,
                
                # Metrics
                'accuracy': result.accuracy,
                'f1_score': result.f1_score,
                'avg_epoch_time': result.avg_epoch_time,
                'total_training_time': result.total_training_time,
                'final_loss': result.final_loss,
                
                # System info
                'platform': self.system_info.get('platform', 'unknown'),
                'cpu_count': self.system_info.get('cpu', {}).get('cpu_count', 'unknown'),
                'memory_total_gb': round(
                    self.system_info.get('memory', {}).get('total', 0) / (1024**3), 2
                )
            }
            all_data.append(result_data)
        
        # Save batch results to CSV
        self._save_batch_to_csv(all_data, csv_path)
        
        # Save detailed batch JSON
        json_data = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'experiments': []
        }
        
        for i, result in enumerate(results):
            exp_data = all_data[i].copy()
            exp_data['loss_history'] = result.loss_history
            json_data['experiments'].append(exp_data)
        
        json_path = self.results_dir / 'metrics' / f'batch_results_{batch_id}.json'
        self._save_to_json(json_data, json_path)
        
        return str(csv_path)
    
    def _save_to_csv(self, data: Dict[str, Any], file_path: Path):
        """Save single experiment data to CSV."""
        fieldnames = [
            'experiment_id', 'timestamp',
            'architecture', 'activation', 'optimizer', 'sequence_length',
            'gradient_clipping', 'learning_rate', 'batch_size', 'epochs', 'dropout',
            'accuracy', 'f1_score', 'avg_epoch_time', 'total_training_time', 'final_loss',
            'platform', 'cpu_count', 'memory_total_gb'
        ]
        
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
    
    def _save_batch_to_csv(self, data_list: List[Dict[str, Any]], file_path: Path):
        """Save multiple experiment data to CSV."""
        if not data_list:
            return
        
        fieldnames = list(data_list[0].keys())
        
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
    
    def _save_to_json(self, data: Dict[str, Any], file_path: Path):
        """Save data to JSON format."""
        with open(file_path, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2, default=str)
    
    def load_experiment_results(self, batch_id: str = None) -> List[Dict[str, Any]]:
        """
        Load experiment results from CSV files.
        
        Args:
            batch_id: Optional batch ID to load specific batch
            
        Returns:
            List[Dict[str, Any]]: List of experiment result dictionaries
        """
        results = []
        metrics_dir = self.results_dir / 'metrics'
        
        if batch_id:
            # Load specific batch
            csv_path = metrics_dir / f'batch_results_{batch_id}.csv'
            if csv_path.exists():
                results.extend(self._load_csv(csv_path))
        else:
            # Load all CSV files
            for csv_file in metrics_dir.glob('*.csv'):
                results.extend(self._load_csv(csv_file))
        
        return results
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from a CSV file."""
        results = []
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert numeric fields
                    numeric_fields = [
                        'sequence_length', 'learning_rate', 'batch_size', 'epochs', 'dropout',
                        'accuracy', 'f1_score', 'avg_epoch_time', 'total_training_time', 
                        'final_loss', 'cpu_count', 'memory_total_gb'
                    ]
                    
                    for field in numeric_fields:
                        if field in row and row[field]:
                            try:
                                row[field] = float(row[field])
                            except ValueError:
                                pass  # Keep as string if conversion fails
                    
                    # Convert boolean fields
                    if 'gradient_clipping' in row:
                        row['gradient_clipping'] = row['gradient_clipping'].lower() == 'true'
                    
                    results.append(row)
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {e}")
        
        return results
    
    def create_summary_report(
        self,
        results: List[Dict[str, Any]] = None,
        output_file: str = None
    ) -> str:
        """
        Create a summary report of all experiments.
        
        Args:
            results: Optional list of results to summarize (loads all if None)
            output_file: Optional output file name
            
        Returns:
            str: Path to the summary report file
        """
        if results is None:
            results = self.load_experiment_results()
        
        if not results:
            print("No results found to summarize")
            return ""
        
        if output_file is None:
            output_file = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report_path = self.results_dir / 'logs' / output_file
        
        with open(report_path, 'w') as f:
            f.write("Sentiment Classification Experiment Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Experiments: {len(results)}\n\n")
            
            # System information
            f.write("System Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Platform: {self.system_info.get('platform', 'unknown')}\n")
            f.write(f"CPU Count: {self.system_info.get('cpu', {}).get('cpu_count', 'unknown')}\n")
            memory_gb = round(self.system_info.get('memory', {}).get('total', 0) / (1024**3), 2)
            f.write(f"Memory: {memory_gb} GB\n\n")
            
            # Best performing configurations
            if results:
                best_accuracy = max(results, key=lambda x: x.get('accuracy', 0))
                best_f1 = max(results, key=lambda x: x.get('f1_score', 0))
                fastest = min(results, key=lambda x: x.get('avg_epoch_time', float('inf')))
                
                f.write("Best Performing Configurations:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Highest Accuracy: {best_accuracy.get('accuracy', 0):.4f}\n")
                f.write(f"  Architecture: {best_accuracy.get('architecture', 'unknown')}\n")
                f.write(f"  Activation: {best_accuracy.get('activation', 'unknown')}\n")
                f.write(f"  Optimizer: {best_accuracy.get('optimizer', 'unknown')}\n\n")
                
                f.write(f"Highest F1-Score: {best_f1.get('f1_score', 0):.4f}\n")
                f.write(f"  Architecture: {best_f1.get('architecture', 'unknown')}\n")
                f.write(f"  Activation: {best_f1.get('activation', 'unknown')}\n")
                f.write(f"  Optimizer: {best_f1.get('optimizer', 'unknown')}\n\n")
                
                f.write(f"Fastest Training: {fastest.get('avg_epoch_time', 0):.2f}s/epoch\n")
                f.write(f"  Architecture: {fastest.get('architecture', 'unknown')}\n")
                f.write(f"  Activation: {fastest.get('activation', 'unknown')}\n")
                f.write(f"  Optimizer: {fastest.get('optimizer', 'unknown')}\n\n")
            
            # Configuration statistics
            architectures = {}
            activations = {}
            optimizers = {}
            
            for result in results:
                arch = result.get('architecture', 'unknown')
                act = result.get('activation', 'unknown')
                opt = result.get('optimizer', 'unknown')
                
                architectures[arch] = architectures.get(arch, []) + [result.get('accuracy', 0)]
                activations[act] = activations.get(act, []) + [result.get('accuracy', 0)]
                optimizers[opt] = optimizers.get(opt, []) + [result.get('accuracy', 0)]
            
            f.write("Average Performance by Configuration:\n")
            f.write("-" * 35 + "\n")
            
            f.write("Architectures:\n")
            for arch, accs in architectures.items():
                f.write(f"  {arch}: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
            
            f.write("\nActivation Functions:\n")
            for act, accs in activations.items():
                f.write(f"  {act}: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
            
            f.write("\nOptimizers:\n")
            for opt, accs in optimizers.items():
                f.write(f"  {opt}: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        
        return str(report_path)
    
    def get_results_directory(self) -> str:
        """Get the results directory path."""
        return str(self.results_dir)
    
    def cleanup_old_results(self, days_old: int = 30):
        """
        Clean up result files older than specified days.
        
        Args:
            days_old: Number of days after which to delete files
        """
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for file_path in self.results_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    print(f"Deleted old result file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import Union


class VisualizationUtilities:
    """
    Visualization utilities for experimental results.
    
    Implements accuracy/F1 vs sequence length plots, creates training loss
    progression plots for best/worst models, and adds comparison charts
    for different configurations.
    """
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize visualization utilities.
        
        Args:
            results_dir: Directory to save plots (default: 'results')
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_metrics_vs_sequence_length(
        self,
        results: List[Dict[str, Any]],
        metric: str = 'accuracy',
        group_by: str = 'architecture',
        save_path: str = None
    ) -> str:
        """
        Plot accuracy/F1-score vs sequence length for different configurations.
        
        Args:
            results: List of experiment result dictionaries
            metric: Metric to plot ('accuracy' or 'f1_score')
            group_by: Configuration parameter to group by ('architecture', 'activation', 'optimizer')
            save_path: Optional custom save path
            
        Returns:
            str: Path to saved plot
        """
        if not results:
            raise ValueError("No results provided for plotting")
        
        # Prepare data for plotting
        plot_data = {}
        sequence_lengths = sorted(list(set(r.get('sequence_length', 0) for r in results)))
        
        for result in results:
            seq_len = result.get('sequence_length', 0)
            group_value = result.get(group_by, 'unknown')
            metric_value = result.get(metric, 0)
            
            if group_value not in plot_data:
                plot_data[group_value] = {sl: [] for sl in sequence_lengths}
            
            if seq_len in plot_data[group_value]:
                plot_data[group_value][seq_len].append(metric_value)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for group_name, group_results in plot_data.items():
            seq_lens = []
            means = []
            stds = []
            
            for seq_len in sequence_lengths:
                if seq_len in group_results and group_results[seq_len]:
                    seq_lens.append(seq_len)
                    values = group_results[seq_len]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
            
            if seq_lens:
                ax.errorbar(seq_lens, means, yerr=stds, marker='o', linewidth=2,
                           markersize=8, capsize=5, label=group_name)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Sequence Length by {group_by.title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sequence_lengths)
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f'{metric}_vs_sequence_length_by_{group_by}.png'
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_training_loss_progression(
        self,
        results: List[ExperimentResult],
        num_models: int = 5,
        criteria: str = 'best',
        save_path: str = None
    ) -> str:
        """
        Plot training loss progression for best/worst performing models.
        
        Args:
            results: List of ExperimentResult objects with loss history
            num_models: Number of models to plot
            criteria: 'best' or 'worst' models based on final accuracy
            save_path: Optional custom save path
            
        Returns:
            str: Path to saved plot
        """
        if not results:
            raise ValueError("No results provided for plotting")
        
        # Sort results by accuracy
        sorted_results = sorted(results, key=lambda x: x.accuracy, reverse=(criteria == 'best'))
        selected_results = sorted_results[:num_models]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, result in enumerate(selected_results):
            if result.loss_history:
                epochs = range(1, len(result.loss_history) + 1)
                label = (f"{result.config.architecture}_{result.config.activation}_"
                        f"{result.config.optimizer} (Acc: {result.accuracy:.3f})")
                ax.plot(epochs, result.loss_history, marker='o', linewidth=2,
                       markersize=4, label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title(f'Training Loss Progression - {criteria.title()} {num_models} Models')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f'loss_progression_{criteria}_{num_models}_models.png'
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_configuration_comparison(
        self,
        results: List[Dict[str, Any]],
        config_param: str,
        metrics: List[str] = None,
        save_path: str = None
    ) -> str:
        """
        Create comparison charts for different configurations.
        
        Args:
            results: List of experiment result dictionaries
            config_param: Configuration parameter to compare ('architecture', 'activation', 'optimizer')
            metrics: List of metrics to compare (default: ['accuracy', 'f1_score'])
            save_path: Optional custom save path
            
        Returns:
            str: Path to saved plot
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_score']
        
        if not results:
            raise ValueError("No results provided for plotting")
        
        # Prepare data
        config_data = {}
        for result in results:
            config_value = result.get(config_param, 'unknown')
            if config_value not in config_data:
                config_data[config_value] = {metric: [] for metric in metrics}
            
            for metric in metrics:
                config_data[config_value][metric].append(result.get(metric, 0))
        
        # Create subplots for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for box plot
            config_names = list(config_data.keys())
            metric_values = [config_data[config][metric] for config in config_names]
            
            # Create box plot
            box_plot = ax.boxplot(metric_values, labels=config_names, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{metric.replace("_", " ").title()} by {config_param.title()}')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            # Add mean values as text
            for j, config in enumerate(config_names):
                mean_val = np.mean(config_data[config][metric])
                ax.text(j + 1, mean_val, f'{mean_val:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Performance Comparison by {config_param.title()}', fontsize=16)
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f'comparison_by_{config_param}.png'
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_performance_heatmap(
        self,
        results: List[Dict[str, Any]],
        x_param: str,
        y_param: str,
        metric: str = 'accuracy',
        save_path: str = None
    ) -> str:
        """
        Create a heatmap showing performance across two configuration parameters.
        
        Args:
            results: List of experiment result dictionaries
            x_param: Parameter for x-axis ('sequence_length', 'learning_rate', etc.)
            y_param: Parameter for y-axis ('architecture', 'activation', etc.)
            metric: Metric to visualize ('accuracy', 'f1_score')
            save_path: Optional custom save path
            
        Returns:
            str: Path to saved plot
        """
        if not results:
            raise ValueError("No results provided for plotting")
        
        # Prepare data for heatmap
        x_values = sorted(list(set(r.get(x_param, 'unknown') for r in results)))
        y_values = sorted(list(set(r.get(y_param, 'unknown') for r in results)))
        
        # Create matrix
        heatmap_data = np.zeros((len(y_values), len(x_values)))
        count_data = np.zeros((len(y_values), len(x_values)))
        
        for result in results:
            x_val = result.get(x_param, 'unknown')
            y_val = result.get(y_param, 'unknown')
            metric_val = result.get(metric, 0)
            
            if x_val in x_values and y_val in y_values:
                x_idx = x_values.index(x_val)
                y_idx = y_values.index(y_val)
                heatmap_data[y_idx, x_idx] += metric_val
                count_data[y_idx, x_idx] += 1
        
        # Average the values
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_data = np.divide(heatmap_data, count_data, 
                                   out=np.zeros_like(heatmap_data), where=count_data!=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(x_values)), max(6, len(y_values))))
        
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(x_values)))
        ax.set_yticks(range(len(y_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticklabels(y_values)
        
        # Rotate x labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.replace('_', ' ').title())
        
        # Add text annotations
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if count_data[i, j] > 0:
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                 ha="center", va="center", color="white", fontweight='bold')
        
        ax.set_title(f'{metric.replace("_", " ").title()} Heatmap: {y_param.title()} vs {x_param.title()}')
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f'heatmap_{metric}_{y_param}_vs_{x_param}.png'
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_training_time_analysis(
        self,
        results: List[Dict[str, Any]],
        save_path: str = None
    ) -> str:
        """
        Create plots analyzing training time across different configurations.
        
        Args:
            results: List of experiment result dictionaries
            save_path: Optional custom save path
            
        Returns:
            str: Path to saved plot
        """
        if not results:
            raise ValueError("No results provided for plotting")
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Training time vs Architecture
        arch_times = {}
        for result in results:
            arch = result.get('architecture', 'unknown')
            time_val = result.get('avg_epoch_time', 0)
            if arch not in arch_times:
                arch_times[arch] = []
            arch_times[arch].append(time_val)
        
        arch_names = list(arch_times.keys())
        arch_values = [arch_times[arch] for arch in arch_names]
        ax1.boxplot(arch_values, labels=arch_names)
        ax1.set_title('Training Time by Architecture')
        ax1.set_ylabel('Avg Epoch Time (s)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Training time vs Sequence Length
        seq_times = {}
        for result in results:
            seq_len = result.get('sequence_length', 0)
            time_val = result.get('avg_epoch_time', 0)
            if seq_len not in seq_times:
                seq_times[seq_len] = []
            seq_times[seq_len].append(time_val)
        
        seq_lens = sorted(seq_times.keys())
        seq_means = [np.mean(seq_times[sl]) for sl in seq_lens]
        seq_stds = [np.std(seq_times[sl]) for sl in seq_lens]
        
        ax2.errorbar(seq_lens, seq_means, yerr=seq_stds, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Training Time vs Sequence Length')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Avg Epoch Time (s)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy vs Training Time scatter
        accuracies = [r.get('accuracy', 0) for r in results]
        times = [r.get('avg_epoch_time', 0) for r in results]
        architectures = [r.get('architecture', 'unknown') for r in results]
        
        # Color by architecture
        unique_archs = list(set(architectures))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_archs)))
        arch_color_map = {arch: colors[i] for i, arch in enumerate(unique_archs)}
        
        for arch in unique_archs:
            arch_acc = [acc for acc, a in zip(accuracies, architectures) if a == arch]
            arch_times = [time for time, a in zip(times, architectures) if a == arch]
            ax3.scatter(arch_times, arch_acc, label=arch, alpha=0.7, s=50)
        
        ax3.set_title('Accuracy vs Training Time')
        ax3.set_xlabel('Avg Epoch Time (s)')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Total training time distribution
        total_times = [r.get('total_training_time', 0) for r in results]
        ax4.hist(total_times, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_title('Total Training Time Distribution')
        ax4.set_xlabel('Total Training Time (s)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Training Time Analysis', fontsize=16)
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / 'training_time_analysis.png'
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_comprehensive_report_plots(
        self,
        results: Union[List[Dict[str, Any]], List[ExperimentResult]],
        report_name: str = None
    ) -> List[str]:
        """
        Create a comprehensive set of plots for experimental results.
        
        Args:
            results: List of experiment results (dict or ExperimentResult objects)
            report_name: Optional name for the report (used in filenames)
            
        Returns:
            List[str]: List of paths to generated plots
        """
        if not results:
            return []
        
        # Convert ExperimentResult objects to dictionaries if needed
        if isinstance(results[0], ExperimentResult):
            dict_results = []
            for result in results:
                dict_result = {
                    'architecture': result.config.architecture,
                    'activation': result.config.activation,
                    'optimizer': result.config.optimizer,
                    'sequence_length': result.config.sequence_length,
                    'gradient_clipping': result.config.gradient_clipping,
                    'learning_rate': result.config.learning_rate,
                    'accuracy': result.accuracy,
                    'f1_score': result.f1_score,
                    'avg_epoch_time': result.avg_epoch_time,
                    'total_training_time': result.total_training_time,
                    'final_loss': result.final_loss
                }
                dict_results.append(dict_result)
        else:
            dict_results = results
        
        plot_paths = []
        prefix = f"{report_name}_" if report_name else ""
        
        try:
            # 1. Accuracy vs Sequence Length by Architecture
            path = self.plot_metrics_vs_sequence_length(
                dict_results, 'accuracy', 'architecture',
                self.plots_dir / f'{prefix}accuracy_vs_seq_len_by_arch.png'
            )
            plot_paths.append(path)
            
            # 2. F1-Score vs Sequence Length by Architecture
            path = self.plot_metrics_vs_sequence_length(
                dict_results, 'f1_score', 'architecture',
                self.plots_dir / f'{prefix}f1_vs_seq_len_by_arch.png'
            )
            plot_paths.append(path)
            
            # 3. Configuration comparisons
            for config_param in ['architecture', 'activation', 'optimizer']:
                path = self.plot_configuration_comparison(
                    dict_results, config_param,
                    save_path=self.plots_dir / f'{prefix}comparison_by_{config_param}.png'
                )
                plot_paths.append(path)
            
            # 4. Performance heatmaps
            path = self.plot_performance_heatmap(
                dict_results, 'sequence_length', 'architecture', 'accuracy',
                self.plots_dir / f'{prefix}heatmap_accuracy_arch_vs_seq.png'
            )
            plot_paths.append(path)
            
            # 5. Training time analysis
            path = self.plot_training_time_analysis(
                dict_results,
                self.plots_dir / f'{prefix}training_time_analysis.png'
            )
            plot_paths.append(path)
            
            # 6. Loss progression (if ExperimentResult objects available)
            if isinstance(results[0], ExperimentResult):
                path = self.plot_training_loss_progression(
                    results, 5, 'best',
                    self.plots_dir / f'{prefix}loss_progression_best.png'
                )
                plot_paths.append(path)
                
                path = self.plot_training_loss_progression(
                    results, 5, 'worst',
                    self.plots_dir / f'{prefix}loss_progression_worst.png'
                )
                plot_paths.append(path)
            
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        return plot_paths