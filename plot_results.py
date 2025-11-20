import pandas as pd
import matplotlib.pyplot as plt
import os
from src.config import PLOTS_DIR, RESULTS_DIR

def plot_accuracy_f1_vs_seq(results_csv=os.path.join(RESULTS_DIR, "experiments_summary.csv")):
    df = pd.read_csv(results_csv)
    #Group by seq length and compute mean (for each optimizer)
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    for opt in df['Optimizer'].unique():
        sub = df[df['Optimizer']==opt]
        sub_sorted = sub.sort_values('Seq_Length')
        ax.plot(sub_sorted['Seq_Length'], sub_sorted['Accuracy'], marker='o', label=f"{opt} - Acc")
        ax.plot(sub_sorted['Seq_Length'], sub_sorted['F1'], marker='x', linestyle='--', label=f"{opt} - F1")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and F1 vs Sequence Length")
    ax.legend()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "accuracy_f1_vs_seq_length.png")
    plt.savefig(out_path)
    print("Saved:", out_path)

def plot_loss_curves(results_csv=os.path.join(RESULTS_DIR, "experiments_summary.csv")):
    df = pd.read_csv(results_csv)
    #picking best and worst by F1
    best_row = df.loc[df['F1'].idxmax()]
    worst_row = df.loc[df['F1'].idxmin()]

    import ast
    best_hist = ast.literal_eval(best_row['History_Val_Loss']) if isinstance(best_row['History_Val_Loss'], str) else best_row['History_Val_Loss']
    worst_hist = ast.literal_eval(worst_row['History_Val_Loss']) if isinstance(worst_row['History_Val_Loss'], str) else worst_row['History_Val_Loss']

    plt.figure(figsize=(8,5))
    plt.plot(best_hist, label=f"Best: {best_row['Optimizer']} seq{best_row['Seq_Length']} (F1={best_row['F1']:.3f})")
    plt.plot(worst_hist, label=f"Worst: {worst_row['Optimizer']} seq{worst_row['Seq_Length']} (F1={worst_row['F1']:.3f})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Training Loss (Best vs Worst)")
    plt.legend()
    out_path = os.path.join(PLOTS_DIR, "training_loss_best_worst.png")
    plt.savefig(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    plot_accuracy_f1_vs_seq()
    plot_loss_curves()