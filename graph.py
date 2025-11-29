import os
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = "graphs"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_last_50_by_day(pred_close, true_close, save_prefix="day"):
    total = len(pred_close)
    num_to_save = min(50, total)

    start = total - num_to_save

    pred_last = pred_close[start:total]   
    true_last = true_close[start:total]   

    for day in range(5):
        pred = pred_last[:, day]          
        true = true_last[:, day]          

        save_path = os.path.join(SAVE_DIR, f"{save_prefix}{day+1}.png")

        plt.figure(figsize=(10, 4))
        plt.plot(true, label="True", linewidth=2)
        plt.plot(pred, label="Pred", linewidth=2)
        plt.title(f"Day {day+1} Prediction (Last 50 Samples)")
        plt.xlabel("Sample Index (Most Recent 50)")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
