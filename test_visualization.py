import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score
import os

from dataset import create_kospi_datasets
from model import CNNLSTM

MODEL_PATH = "best_cnn_lstm_kospi.pth"
CSV_PATH = ".\KOSPI_merged_dataset_ffill.csv" 
SEQ_LEN = 30
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "presentation_images"

FEATURE_COLS = [
    "KOSPI_Open", "KOSPI_High", "KOSPI_Low", "KOSPI_Close",
    "SP500_Close", "USD_KRW_Close", "WTI_OIL_Close", "GOLD_Close",
    "KOSPI_Return", "KOSPI_Volatility20",
    "KOSPI_Volume", "KOSPI_MA10", "KOSPI_MA30", "KOSPI_MA60"
]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def load_data_and_model():
    print(f"데이터 로드 중: {CSV_PATH}")
    train_ds, val_ds, test_ds = create_kospi_datasets(CSV_PATH, seq_len=SEQ_LEN)
    
    train_X_list = [x.numpy() for x, _ in train_ds]
    train_X_all = np.concatenate(train_X_list, axis=0)
    scaler_x = MinMaxScaler()
    scaler_x.fit(train_X_all)

    train_y_list = [y.numpy() for _, y in train_ds]
    train_y_all = np.array(train_y_list)
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_y_all) 

    model = CNNLSTM(num_features=len(FEATURE_COLS)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("모델 로드 완료.")
    else:
        print("모델 파일 없음.")
        exit()
    
    model.eval()
    return test_ds, model, scaler_x, scaler_y

def run_inference(dataset, model, scaler_x, scaler_y):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_targets, all_last_closes = [], [], []

    print("테스트 예측 실행 중...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            B, T, F = x.shape
            
            x_reshaped = x.cpu().numpy().reshape(-1, F)
            x_scaled = scaler_x.transform(x_reshaped).reshape(B, T, F)
            x_tensor = torch.FloatTensor(x_scaled).to(DEVICE)

            preds_scaled = model(x_tensor).cpu().numpy()
            preds_diff = scaler_y.inverse_transform(preds_scaled)
            targets_diff = y.numpy()
            last_close = x.cpu().numpy()[:, -1, 3] 
            
            all_preds.append(preds_diff)
            all_targets.append(targets_diff)
            all_last_closes.append(last_close)

    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_last_closes)

def restore_price(last_close, diffs):
    prices = []
    for i in range(len(last_close)):
        base = last_close[i]
        seq_price = []
        for j in range(5):
            base += diffs[i][j]
            seq_price.append(base)
        prices.append(seq_price)
    return np.array(prices)

def plot_scatter_by_step(true_prices, pred_prices):
    for i in range(5):
        t, p = true_prices[:, i], pred_prices[:, i]
        r2 = r2_score(t, p)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(t, p, alpha=0.4, s=10, color='blue')
        min_v, max_v = min(t.min(), p.min()), max(t.max(), p.max())
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Fit')
        
        plt.title(f"Scatter Plot - Day {i+1} (R2: {r2:.4f})", fontsize=14, fontweight='bold')
        plt.xlabel("True Price")
        plt.ylabel("Predicted Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/1_scatter_day_{i+1}.png")
        plt.close()
    print("1. 산점도 저장 완료.")

def plot_error_distribution_by_step(true_prices, pred_prices):
    errors = pred_prices - true_prices 
    min_err, max_err = errors.min(), errors.max()
    
    for i in range(5):
        day_error = errors[:, i]
        rmse = np.sqrt(np.mean(day_error**2))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(day_error, kde=True, bins=40, color='purple')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        
        plt.title(f"Error Distribution - Day {i+1} (RMSE: {rmse:.1f})", fontsize=14, fontweight='bold')
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.xlim(min_err, max_err)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/2_error_dist_day_{i+1}.png")
        plt.close()
    print("2. 에러 분포 저장 완료.")

def plot_case_study(true_prices, pred_prices):
    returns = (true_prices[:, -1] - true_prices[:, 0]) / true_prices[:, 0]
    rise_idx = np.argmax(returns)
    fall_idx = np.argmin(returns)
    flat_idx = np.argmin(np.abs(returns))
    
    indices = [('Rapid Rise', rise_idx), ('Rapid Fall', fall_idx), ('Sideways', flat_idx)]
    
    plt.figure(figsize=(18, 5))
    for i, (label, idx) in enumerate(indices):
        plt.subplot(1, 3, i+1)
        rmse = np.sqrt(mean_squared_error(true_prices[idx], pred_prices[idx]))
        
        plt.plot(range(1, 6), true_prices[idx], 'b-o', label='Actual', linewidth=2)
        plt.plot(range(1, 6), pred_prices[idx], 'r--x', label='Predicted', linewidth=2)
        plt.title(f"{label}\nRMSE: {rmse:.2f}")
        plt.xlabel("Day (1~5)")
        plt.ylabel("Price")
        plt.xticks(range(1, 6))
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/3_case_studies.png")
    plt.close()
    print("3. 케이스 스터디 저장 완료.")

def plot_full_test_tracking(true_prices, pred_prices):
    time_steps = np.arange(len(true_prices))
    
    for i in range(5):
        plt.figure(figsize=(12, 6)) 
        
        t_data = true_prices[:, i]
        p_data = pred_prices[:, i]
        
        plt.plot(time_steps, t_data, label=f'Actual (Day {i+1})', color='black', alpha=0.7, linewidth=1)
        plt.plot(time_steps, p_data, label=f'Predicted (Day {i+1})', color='red', alpha=0.6, linewidth=1)
        
        plt.title(f"Full Test Set Tracking - Day {i+1}", fontsize=14, fontweight='bold')
        plt.xlabel("Test Set Sample Index")
        plt.ylabel("Price")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/4_full_tracking_day_{i+1}.png")
        plt.close()
    print("4. 전체 추적 그래프 저장 완료.")

def plot_step_wise_performance(true_prices, pred_prices):
    rmses = []
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
    for i in range(5):
        mse = mean_squared_error(true_prices[:, i], pred_prices[:, i])
        rmses.append(np.sqrt(mse))
        
    plt.figure(figsize=(10, 6))
    bars = plt.bar(days, rmses, color='cornflowerblue', edgecolor='black', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}', 
                 ha='center', va='bottom', fontweight='bold')
    plt.title("RMSE by Prediction Horizon")
    plt.ylabel("RMSE")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f"{SAVE_DIR}/5_step_wise_rmse.png")
    plt.close()
    print("5. 단계별 성능 저장 완료.")

def plot_direction_accuracy(last_closes, true_prices, pred_prices):
    true_diff = true_prices[:, 0] - last_closes
    pred_diff = pred_prices[:, 0] - last_closes
    true_dir = (true_diff >= 0).astype(int)
    pred_dir = (pred_diff >= 0).astype(int)
    
    cm = confusion_matrix(true_dir, pred_dir)
    acc = accuracy_score(true_dir, pred_dir)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Down', 'Pred Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    plt.title(f"Directional Accuracy (Day 1): {acc*100:.2f}%")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"{SAVE_DIR}/6_direction_accuracy.png")
    plt.close()
    print("6. 방향성 정확도 저장 완료.")

def plot_error_vs_volatility(true_prices, pred_prices):
    volatility = np.std(true_prices, axis=1)
    mse_per_sample = np.mean((true_prices - pred_prices)**2, axis=1)
    rmse_per_sample = np.sqrt(mse_per_sample)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(volatility, rmse_per_sample, alpha=0.4, color='green')
    
    z = np.polyfit(volatility, rmse_per_sample, 1)
    p = np.poly1d(z)
    plt.plot(volatility, p(volatility), "r--", linewidth=2, label='Trend')
    
    plt.title("Error vs. Market Volatility")
    plt.xlabel("Volatility (Std Dev of True Price)")
    plt.ylabel("Prediction Error (RMSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/7_error_vs_volatility.png")
    plt.close()
    print("7. 변동성 분석 저장 완료.")

if __name__ == "__main__":
    test_ds, model, sx, sy = load_data_and_model()
    print(f"테스트 데이터셋 총 샘플 수: {len(test_ds)}개")
    preds_diff, targets_diff, last_closes = run_inference(test_ds, model, sx, sy)
    
    pred_prices = restore_price(last_closes, preds_diff)
    true_prices = restore_price(last_closes, targets_diff)
    
    print(f"\n--- 저장 경로: {SAVE_DIR} ---")
    plot_scatter_by_step(true_prices, pred_prices)
    plot_error_distribution_by_step(true_prices, pred_prices)
    plot_case_study(true_prices, pred_prices)
    plot_full_test_tracking(true_prices, pred_prices)
    plot_step_wise_performance(true_prices, pred_prices)
    plot_direction_accuracy(last_closes, true_prices, pred_prices)
    plot_error_vs_volatility(true_prices, pred_prices)
    
    print("\n모든 작업 완료.")