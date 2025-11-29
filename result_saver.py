import os
import json

SAVE_DIR = "hyperparameter_tuning"


def get_next_filename():
    os.makedirs(SAVE_DIR, exist_ok=True)

    existing = [
        int(f.split(".")[0]) for f in os.listdir(SAVE_DIR)
        if f.endswith(".txt") and f.split(".")[0].isdigit()
    ]

    next_index = (max(existing) + 1) if existing else 1
    return os.path.join(SAVE_DIR, f"{next_index}.txt")


def save_result(config, test_mse, test_rmse, pred_close, true_close):
    filepath = get_next_filename()

    lines = []
    lines.append("=== CONFIG.json ===\n")
    lines.append(json.dumps(config, indent=4))
    lines.append("\n\n=== TEST RESULTS ===\n")
    lines.append(f"Test MSE (scaled): {test_mse:.6f}\n")
    lines.append(f"Test RMSE (scaled): {test_rmse:.6f}\n")

    lines.append("\n=== 최근 5 일 예측 ===\n")

    num_to_show = 5
    N = len(pred_close)
    start = max(0, N - num_to_show)

    for idx in range(N - 1, start - 1, -1):
        lines.append(f"\nIndex {idx}:\n")
        for day in range(5):
            pred_v = pred_close[idx][day]
            true_v = true_close[idx][day]
            diff_v = pred_v - true_v
            lines.append(
                f" Day{day+1}  Pred={pred_v:8.2f}  True={true_v:8.2f}  Diff={diff_v:8.2f}\n"
            )

    lines.append("\데이터 저장 완료\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"결과 저장 완료 {filepath}")
