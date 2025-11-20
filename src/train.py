import time
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
from src.utils import set_seed, mkdirs, now_str
from src.evaluate import evaluate_model

def get_optimizer(opt_name, model, lr=1e-3):
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer: " + opt_name)

def train_one_experiment(cfg, train_loader, val_loader, model, criterion, optimizer_name,
                         device="cpu", grad_clip=False, epochs=10, save_dir="models"):
    set_seed(cfg.get("seed", 42))
    model.to(device)
    optimizer = get_optimizer(optimizer_name, model, lr=cfg.get("lr", 1e-3))
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "epoch_time": []}

    mkdirs(save_dir)
    best_val_f1 = -1.0
    best_ckpt = None

    for epoch in range(1, epochs+1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        steps = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device).squeeze(1)
            optimizer.zero_grad()
            probs, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            steps += 1

        epoch_time = time.time() - t0
        avg_train_loss = running_loss / max(1, steps)

        #Validation
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device=device)
        val_loss = 0.0
        #compute validation loss for logging
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).squeeze(1)
                _, logits = model(x)
                val_loss += criterion(logits, y).item()
        val_loss /= len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["epoch_time"].append(epoch_time)

        #Save best
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_ckpt = os.path.join(save_dir, f"best_{now_str()}.pth")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
                "history": history
            }, best_ckpt)

    return history, best_ckpt
