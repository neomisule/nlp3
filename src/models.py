import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRNNModel(nn.Module, ABC):
    #Abstract base class for all RNN model architectures.
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super(BaseRNNModel, self).__init__()
        
        #store configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.activation_name = activation
        
        #Embedding - 100 dimensions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        #Activation function mapping
        self.activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        
        self.activation = self.activation_functions[activation]
        
        #Binary classification output layer with sigmoid
        self.output_layer = nn.Linear(hidden_size, 1)
        self.output_activation = nn.Sigmoid()
        
    @abstractmethod
    def _build_rnn_layers(self):
       #Build the RNN layers specific to each architecture
        pass
    
    def get_config(self) -> Dict[str, Any]:
        #Return model configuration as dictionary
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_prob,
            'activation': self.activation_name
        }
    
    def count_parameters(self) -> int:
        #Count total number of trainable parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class SimpleRNN(BaseRNNModel):
    #Basic RNN model with 2 hidden layers for sentiment classification
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super(SimpleRNN, self).__init__(
            vocab_size, embedding_dim, hidden_size, num_layers, dropout, activation
        )
        self._build_rnn_layers()
    
    def _build_rnn_layers(self):
        #Build the RNN layers with specified configuration
        #RNN layers with dropout (except for single layer)
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            batch_first=True,
            nonlinearity='tanh'  #RNN uses tanh internally, we apply activation after
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #embedding layer
        embedded = self.embedding(x)  #(batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        #RNN layers
        rnn_out, hidden = self.rnn(embedded)  # (batch_size, seq_len, hidden_size)
        
        #using last output for classification
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        #applying activation function
        activated = self.activation(last_output)
        activated = self.dropout(activated)
        
        #output layer with sigmoid
        output = self.output_layer(activated)  # (batch_size, 1)
        output = self.output_activation(output)
        
        return output

class LSTMModel(BaseRNNModel):
    #LSTM model with 2 hidden layers for sentiment classification    
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super(LSTMModel, self).__init__(
            vocab_size, embedding_dim, hidden_size, num_layers, dropout, activation
        )
        self._build_rnn_layers()
        self._initialize_lstm_weights()
    
    def _build_rnn_layers(self):
        #Build the LSTM layers with specified configuration
        # LSTM layers with dropout (except for single layer)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            batch_first=True
        )
    
    def _initialize_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Bias initialization
                param.data.fill_(0.)
                # Set forget gate bias to 1 (common LSTM practice)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM layers
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply activation function
        activated = self.activation(last_output)
        activated = self.dropout(activated)
        
        # Output layer with sigmoid
        output = self.output_layer(activated)  # (batch_size, 1)
        output = self.output_activation(output)
        
        return output

class BidirectionalLSTMModel(BaseRNNModel):
    #Bidirectional LSTM model with 2 hidden layers for sentiment classification.
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        #Initialize Bidirectional LSTM model.
        
        super(BidirectionalLSTMModel, self).__init__(
            vocab_size, embedding_dim, hidden_size, num_layers, dropout, activation
        )
        self._build_rnn_layers()
        self._initialize_lstm_weights()
    
    def _build_rnn_layers(self):
        #Build the bidirectional LSTM layers with specified configuration
        # Bidirectional LSTM layers with dropout (except for single layer)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Update output layer to handle concatenated bidirectional output
        # Bidirectional LSTM outputs 2 * hidden_size
        self.output_layer = nn.Linear(self.hidden_size * 2, 1)
    
    def _initialize_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Bias initialization
                param.data.fill_(0.)
                # Set forget gate bias to 1 for both directions
                n = param.size(0)
                # For bidirectional LSTM, we have biases for both directions
                # Each direction has 4 gates, so we need to handle both
                forget_gate_start_1 = n // 8
                forget_gate_end_1 = n // 4
                forget_gate_start_2 = 5 * n // 8
                forget_gate_end_2 = 3 * n // 4
                param.data[forget_gate_start_1:forget_gate_end_1].fill_(1.)
                param.data[forget_gate_start_2:forget_gate_end_2].fill_(1.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Bidirectional LSTM layers
        # Output shape: (batch_size, seq_len, hidden_size * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last output for classification
        # For bidirectional LSTM, we get concatenated forward and backward outputs
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Apply activation function
        activated = self.activation(last_output)
        activated = self.dropout(activated)
        
        # Output layer with sigmoid
        output = self.output_layer(activated)  # (batch_size, 1)
        output = self.output_activation(output)
        
        return output