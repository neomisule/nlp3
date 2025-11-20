This project systematically compares Bidirectional LSTM (Bi-LSTM) models for sentiment classification on the IMDb Movie Review Dataset. The goal is to evaluate how performance is affected by different optimizers, sequence lengths, and the use of gradient clipping.

Core Experiments

- Model: Bidirectional LSTM (with ReLU)
- Optimizers: Adam vs. SGD
- Sequence Lengths: 25, 50, and 100 words
- Stability: Gradient Clipping (On vs. Off)

Setup & Installation

1. Prerequisites:
- Python 3.8+
- Hardware: 4-8 GB RAM and 2+ GB free storage are recommended.

2. Clone & Install:

Bash '''python Clone the repository git clone https://github.com/neomisule/nlp3.git cd nlp3

Create and activate a virtual environment python -m venv venv source, venv/bin/activate # (or venv\Scripts\activate on Windows)

Install dependencies pip install -r req.txt 

Then python runner.py and plot_results.py

3. Get the Dataset:
- Download the IMDb Movie Review Dataset (IMDB_Dataset.csv).
- Place the file inside the data/ directory. '''

How to Run
-	Run All Training Experiments: This script trains a model for every combination of settings defined in src/config.py. Then run python runner.py
-	Visualize Results (Evaluation): After training is complete, generate plots from the saved results. To do this run python plot_results.py

Expected Outputs
-	Logs: results/experiments_summary.csv
  
--	A complete log of all experiments with columns for configuration (Optimizer, Seq Length, etc.) and metrics (Accuracy, F1 Score, Loss History).

- Models: models/

--	All trained model checkpoints (e.g., exp_XYZ _model.pth) are saved here. Each file contains the model weights and final results.

- Plots: results/plots/

--	accuracy_f1_vs_seq_length.png: Bar charts comparing performance.

-- training_loss_best_worst.png: Training curves for the best/worst models
