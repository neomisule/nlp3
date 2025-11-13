from dataclasses import dataclass
from typing import List


#Reproducibility settings
RANDOM_SEED = 42

#data
VOCAB_SIZE = 10000
SEQUENCE_LENGTHS = [25, 50, 100]
TRAIN_TEST_SPLIT = 0.5

#architecture 
EMBEDDING_DIM = 100
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

#configurations
ARCHITECTURES = ['bidirectional_lstm']
ACTIVATION_FUNCTIONS = ['relu']
OPTIMIZERS = ['adam', 'sgd', 'rmsprop']

# File paths
DATA_DIR = 'data'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
SRC_DIR = 'src'


@dataclass
class ExperimentConfig:
    #Config for a single experiment run
    architecture: str
    activation: str
    optimizer: str
    sequence_length: int
    gradient_clipping: bool
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    dropout: float = DROPOUT

@dataclass
class ExperimentResult:
    #Results from a single experiment run
    config: ExperimentConfig
    accuracy: float
    f1_score: float
    avg_epoch_time: float
    total_training_time: float
    final_loss: float
    loss_history: List[float]