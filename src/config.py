from itertools import product

SEED = 42
EMBED_DIM = 100
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 10

MAX_VOCAB_SIZE = 10000
SEQ_LENGTHS = [25, 50, 100]

ACTIVATIONS = {
    "relu": "relu",
    "tanh": "tanh",
    "sigmoid": "sigmoid"
}

ARCHS = ["rnn", "lstm", "bilstm"]

OPTIMIZERS = ["adam", "sgd", "rmsprop"]  

#Gradient clipping options
GRAD_CLIPPING = [False, True]

DEFAULT_ARCH = "bilstm"
DEFAULT_ACT = "relu"
DEFAULT_OPTS = ["adam", "sgd"] 
DEFAULT_SEQ_LEN = [25, 50, 100]
DEFAULT_GRAD_CLIP = [False, True]

#location to save outputs
RESULTS_DIR = "results"
MODELS_DIR = "models"
PLOTS_DIR = f"{RESULTS_DIR}/plots"

DEVICE = "cuda" if False else "cpu"  # default to CPU, change to cuda if available
