"""
- Accuracy and F1 vs Sequence Length (reads `results/experiments_summary.xlsx` or CSV)
- Training Loss vs Epochs for the best and worst models (reads `models/*.pth` for saved `result.loss_history`)
"""
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import ast


def plot_accuracy_f1_vs_seq_len(csv_path, out_path) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    # Support both Excel and CSV input
    if csv_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(csv_path, engine='openpyxl')
    else:
        df = pd.read_csv(csv_path)

    # Ensure numeric seq length
    df['Seq Length'] = pd.to_numeric(df['Seq Length'], errors='coerce')
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    df['F1'] = pd.to_numeric(df['F1'], errors='coerce')

    # Group by sequence length and compute mean (in case multiple runs)
    grouped = df.groupby('Seq Length').agg({'Accuracy': 'mean', 'F1': 'mean'}).reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped['Seq Length'], grouped['Accuracy'], marker='o', label='Accuracy')
    plt.plot(grouped['Seq Length'], grouped['F1'], marker='s', label='F1')
    plt.xlabel('Sequence Length')
    plt.ylabel('Score')
    plt.title('Accuracy and F1 vs Sequence Length')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Accuracy/F1 vs Seq Length plot to {out_path}")

def _parse_loss_history(cell: Any) -> List[float]:

    if isinstance(cell, (list, tuple)):
        return [float(x) for x in cell]
    
    if isinstance(cell, (int, float)):
        return [float(cell)]

    s = str(cell).strip()
    if not s:
        return []

    # Try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple)):
            return [float(x) for x in parsed]
    except Exception:
        pass

    # Try Python literal parsing (safe)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [float(x) for x in parsed]
    except Exception:
        pass

    # As a last resort, try splitting on commas
    try:
        parts = [p.strip() for p in s.strip('[]()').split(',') if p.strip()]
        return [float(p) for p in parts]
    except Exception:
        return []

def plot_training_loss_best_worst(csv_path, out_path) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    # Support both Excel and CSV input
    if csv_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(csv_path, engine='openpyxl')
    else:
        df = pd.read_csv(csv_path)
    records = []
    for idx, row in df.iterrows():
        loss_history = _parse_loss_history(row.get('Loss History'))
        acc = float(row.get('Accuracy'))
        records.append({'index': idx, 'accuracy': acc, 'loss_history': loss_history})

    # Determine best and worst by accuracy
    records_with_acc = [r for r in records if r['accuracy'] is not None]
    best = max(records_with_acc, key=lambda x: x['accuracy'])
    worst = min(records_with_acc, key=lambda x: x['accuracy'])

    plt.figure(figsize=(8, 5))
    epochs_best = list(range(1, len(best['loss_history']) + 1))
    epochs_worst = list(range(1, len(worst['loss_history']) + 1))
    plt.plot(epochs_best, best['loss_history'], marker='o', label=f"Best (idx={best.get('index')}, acc={best.get('accuracy')})")
    plt.plot(epochs_worst, worst['loss_history'], marker='s', label=f"Worst (idx={worst.get('index')}, acc={worst.get('accuracy')})")
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epochs (Best and Worst Models)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Training Loss plot to {out_path}")


def main():
    results_file = Path("results/experiments_summary.xlsx")
    # fall back to csv if xlsx not present
    if not results_file.exists():
        results_file = Path("results/experiments_summary.csv")

    plot_accuracy_f1_vs_seq_len(results_file, "results/plots/accuracy_f1_vs_seq_length.png")
    plot_training_loss_best_worst(results_file, "results/plots/training_loss_best_worst.png")


if __name__ == '__main__':
    main()