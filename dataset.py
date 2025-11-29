import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


class KOSPIDataset(Dataset):
    def __init__(self, csv_path, seq_len=30, use_tomorrow=True):
        """
        csv_path   : KOSPI_merged_dataset_ffill.csv 경로
        seq_len    : 입력 시퀀스 길이 (예: 30일)
        use_tomorrow : True면 '내일 종가' 관련 타깃 사용
        """
        df = pd.read_csv(csv_path)

        if "Date" in df.columns:
            date_col = "Date"
        elif "Unnamed: 0" in df.columns:
            date_col = "Unnamed: 0"
        else:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        df = df.set_index(date_col)
        df = df.sort_index()

        self.feature_cols = [
            "KOSPI_Open",
            "KOSPI_High",
            "KOSPI_Low",
            "KOSPI_Close",
            "SP500_Close",
            "USD_KRW_Close",
            "WTI_OIL_Close",
            "GOLD_Close",
            "KOSPI_Return",
            "KOSPI_Volatility20",
            "KOSPI_Volume",
            "KOSPI_MA10",
            "KOSPI_MA30",
            "KOSPI_MA60"
        ]
        self.target_col = "KOSPI_Close"

        cols_to_num = self.feature_cols + [self.target_col]
        for c in cols_to_num:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=cols_to_num)

        df["Close_shift_1"] = df[self.target_col].shift(-1)
        df["Close_shift_2"] = df[self.target_col].shift(-2)
        df["Close_shift_3"] = df[self.target_col].shift(-3)
        df["Close_shift_4"] = df[self.target_col].shift(-4)
        df["Close_shift_5"] = df[self.target_col].shift(-5)

        df["Δ1"] = df["Close_shift_1"] - df[self.target_col]
        df["Δ2"] = df["Close_shift_2"] - df["Close_shift_1"]
        df["Δ3"] = df["Close_shift_3"] - df["Close_shift_2"]
        df["Δ4"] = df["Close_shift_4"] - df["Close_shift_3"]
        df["Δ5"] = df["Close_shift_5"] - df["Close_shift_4"]

        df = df.dropna(subset=["Δ1", "Δ2", "Δ3", "Δ4", "Δ5"])

        self.seq_len = seq_len
        self.df = df

        self.features = df[self.feature_cols].values
        self.targets = df[["Δ1", "Δ2", "Δ3", "Δ4", "Δ5"]].values

    def __len__(self):
        return len(self.df) - self.seq_len - 4

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


def create_kospi_datasets(
    csv_path,
    seq_len=10,
    use_tomorrow=True,
    train_ratio=0.8,
    val_ratio=0.1
):
    full_dataset = KOSPIDataset(csv_path, seq_len=seq_len, use_tomorrow=use_tomorrow)

    n = len(full_dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_indices = range(0, train_end)
    val_indices   = range(train_end, val_end)
    test_indices  = range(val_end, n)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)
    test_dataset  = Subset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset



""""
KOSPIDataset
-> x: (시퀀스 길이, 데이터 피처) , y: 종가 변화량 (시퀀스 마지막 날 - 시퀀스 마지막 날-1)
seq_len: 시퀀스 길이 (윈도우 크기)
seq_len 이하의 값은 버림
데이터 구조: (N, num_features) N: 전체 데이터의 일, num_features: 피처 수 
"""
