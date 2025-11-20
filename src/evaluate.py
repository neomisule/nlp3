import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            probs, _ = model(x)
            pred = (probs >= 0.5).long().cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(y.long().cpu().numpy().ravel().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    return {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec}
