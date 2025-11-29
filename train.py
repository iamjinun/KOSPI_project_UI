import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset import create_kospi_datasets
from model import CNNLSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import graph
from result_saver import save_result


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


def val_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = criterion(preds, y)
            running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


class ScaledDataset(Dataset):
    def __init__(self, base_dataset, scaler_x, scaler_y):
        self.base_dataset = base_dataset
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        x_scaled = self.scaler_x.transform(x.numpy())
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32)

        y_scaled = self.scaler_y.transform(y.numpy().reshape(1, -1))[0]
        y_scaled = torch.tensor(y_scaled, dtype=torch.float32)

        return x_scaled, y_scaled


def main():

    config = load_config("config.json")
    data_cfg = config["data"]
    train_cfg = config["train"]

    csv_path     = data_cfg["csv_path"]
    seq_len      = data_cfg["seq_len"]
    use_tomorrow = data_cfg["use_tomorrow"]
    train_ratio  = data_cfg["train_ratio"]
    val_ratio    = data_cfg["val_ratio"]

    batch_size    = train_cfg["batch_size"]
    num_epochs    = train_cfg["num_epochs"]
    learning_rate = train_cfg["learning_rate"]
    shuffle_train = train_cfg["shuffle"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset, val_dataset, test_dataset = create_kospi_datasets(
        csv_path=csv_path,
        seq_len=seq_len,
        use_tomorrow=use_tomorrow,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    train_X_list = [x.numpy() for x, _ in train_dataset]
    train_X_all = np.concatenate(train_X_list, axis=0)

    scaler_x = MinMaxScaler()
    scaler_x.fit(train_X_all)

    train_y = np.array([y.numpy() for _, y in train_dataset])
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_y)

    train_dataset = ScaledDataset(train_dataset, scaler_x, scaler_y)
    val_dataset   = ScaledDataset(val_dataset, scaler_x, scaler_y)
    test_dataset  = ScaledDataset(test_dataset, scaler_x, scaler_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNNLSTM(num_features=14).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir="runs/kospi_cnn_lstm")

    best_val_loss = float("inf")
    best_model_path = "best_cnn_lstm_kospi.pth"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = val_one_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch:03d}/{num_epochs}] Train={train_loss:.6f} | Val={val_loss:.6f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(" → New best model saved.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss = val_one_epoch(model, test_loader, criterion, device)
    test_rmse = np.sqrt(test_loss)

    print(f"[TEST] MSE_scaled={test_loss:.6f}")
    print(f"[TEST] RMSE_scaled={test_rmse:.6f}")

    writer.add_scalar("Test/MSE_scaled", test_loss)
    writer.add_scalar("Test/RMSE_scaled", test_rmse)

    model.eval()
    pred_list = []
    true_list = []
    close_today_list = []

    CLOSE_INDEX = 3

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).cpu()

            pred_list.append(preds.numpy())
            true_list.append(y.numpy())

            x_scaled = x.cpu().numpy()
            B, T, F = x_scaled.shape

            x_real_2d = scaler_x.inverse_transform(x_scaled.reshape(-1, F))
            x_real = x_real_2d.reshape(B, T, F)

            close_today_list.append(x_real[:, -1, CLOSE_INDEX])

    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)
    close_today_list = np.concatenate(close_today_list)

    pred_delta = scaler_y.inverse_transform(pred_list)
    true_delta = scaler_y.inverse_transform(true_list)

    pred_close = []
    true_close = []

    for i in range(len(pred_delta)):
        base = close_today_list[i]
        pred_c1 = base + pred_delta[i][0]
        pred_c2 = pred_c1 + pred_delta[i][1]
        pred_c3 = pred_c2 + pred_delta[i][2]
        pred_c4 = pred_c3 + pred_delta[i][3]
        pred_c5 = pred_c4 + pred_delta[i][4]
        pred_close.append([pred_c1, pred_c2, pred_c3, pred_c4, pred_c5])

        true_c1 = base + true_delta[i][0]
        true_c2 = true_c1 + true_delta[i][1]
        true_c3 = true_c2 + true_delta[i][2]
        true_c4 = true_c3 + true_delta[i][3]
        true_c5 = true_c4 + true_delta[i][4]
        true_close.append([true_c1, true_c2, true_c3, true_c4, true_c5])

    pred_close = np.array(pred_close)
    true_close = np.array(true_close)

    print("\n--- 최근 5 일 예측 ---\n")

    num_to_show = 5
    N = len(pred_close)

    start = max(0, N - num_to_show)

    for idx in range(N-1, start-1, -1):
        print(f"\nIndex {idx}:")
        for day in range(5):
            pred_v = pred_close[idx][day]
            true_v = true_close[idx][day]
            diff_v = pred_v - true_v
            print(f" Day{day+1}  Pred={pred_v:8.2f}  True={true_v:8.2f}  Diff={diff_v:8.2f}")

        writer.close()

    graph.plot_last_50_by_day(pred_close, true_close)
    print("그래프 저장 완료")

    save_result(config, test_loss, test_rmse, pred_close, true_close)

if __name__ == "__main__":
    main()
