import os
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.config import DEFAULT_ARCH, DEFAULT_ACT, DEFAULT_OPTS, DEFAULT_SEQ_LEN, DEFAULT_GRAD_CLIP, \
                       EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BATCH_SIZE, EPOCHS, MAX_VOCAB_SIZE, \
                       RESULTS_DIR, MODELS_DIR, PLOTS_DIR, DEVICE, SEED
from src.utils import set_seed, mkdirs, save_json
from src.dataset import IMDBDataset
from src.models import BaseRNNClassifier
from src.train import train_one_experiment
import torch.nn as nn

mkdirs(RESULTS_DIR, MODELS_DIR, PLOTS_DIR)
set_seed(SEED)

#loadin dataset
def run_all(data_dir="data", out_csv=os.path.join(RESULTS_DIR, "experiments_summary.csv")):
    rows = []
    combos = list(itertools.product(DEFAULT_OPTS, DEFAULT_SEQ_LEN, DEFAULT_GRAD_CLIP))
    print(f"Running {len(combos)} experiments (optimizers x seq_length x grad_clip)")

    for opt_name, seq_len, grad_clip in combos:
        print("Experiment:", opt_name, seq_len, grad_clip)
        #data paths
        Xtr = os.path.join(data_dir, f"X_train_seq{seq_len}.npy")
        Xte = os.path.join(data_dir, f"X_test_seq{seq_len}.npy")
        ytr = os.path.join(data_dir, f"y_train_seq{seq_len}.npy")
        yte = os.path.join(data_dir, f"y_test_seq{seq_len}.npy")

        train_ds = IMDBDataset(Xtr, ytr)
        test_ds = IMDBDataset(Xte, yte)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = BaseRNNClassifier(
            arch="bilstm",
            vocab_size=MAX_VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            activation="relu",
            bidirectional=True
        )

        criterion = nn.BCEWithLogitsLoss()
        cfg = {
            "arch": "bilstm",
            "activation": "relu",
            "optimizer": opt_name,
            "seq_len": seq_len,
            "grad_clip": grad_clip,
            "seed": SEED,
            "embed_dim": EMBED_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        }

        history, best_ckpt = train_one_experiment(cfg, train_loader, test_loader, model,
                                                 criterion, opt_name, device=DEVICE,
                                                 grad_clip=grad_clip, epochs=EPOCHS,
                                                 save_dir=MODELS_DIR)

        #summarize results
        last_idx = -1
        row = {
            "Model": "bidirectional_lstm",
            "Activation": "relu",
            "Optimizer": opt_name,
            "Seq_Length": seq_len,
            "Grad_Clipping": grad_clip,
            "Accuracy": history["val_acc"][last_idx],
            "F1": history["val_f1"][last_idx],
            "Epoch_Time(s)": history["epoch_time"][last_idx],
            "Final_Loss": history["val_loss"][last_idx],
            "History_Train_Loss": history["train_loss"],
            "History_Val_Loss": history["val_loss"],
            "Checkpoint": best_ckpt
        }
        rows.append(row)
        # save intermediate CSV
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Done. Results saved to", out_csv)

if __name__ == "__main__":
    run_all()