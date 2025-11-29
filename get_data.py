import yfinance as yf
import pandas as pd
import os

save_path = "."
os.makedirs(save_path, exist_ok=True)

start_date = "2020-01-01"
end_date = '2025-11-28'

tickers = {
    "KOSPI": "^KS11",
    "SP500": "^GSPC",
    "USD_KRW": "KRW=X",
    "WTI_OIL": "CL=F",
    "GOLD": "GC=F"
}

all_data = {}
for name, ticker in tickers.items():
    data = yf.download(ticker, start=start_date, end=end_date)
    all_data[name] = data

full_dates = pd.date_range(start=start_date, end=end_date, freq="D")
full_dates = full_dates[full_dates.weekday < 5]

merged = all_data["KOSPI"][["Open", "High", "Low", "Close"]].copy()
merged = merged.rename(columns={
    "Open":  "KOSPI_Open",
    "High":  "KOSPI_High",
    "Low":   "KOSPI_Low",
    "Close": "KOSPI_Close"
})
merged = merged.reindex(full_dates)

merged["SP500_Close"]    = all_data["SP500"]["Close"]
merged["USD_KRW_Close"]  = all_data["USD_KRW"]["Close"]
merged["WTI_OIL_Close"]  = all_data["WTI_OIL"]["Close"]
merged["GOLD_Close"]     = all_data["GOLD"]["Close"]
merged = merged.reindex(full_dates)

merged["KOSPI_Volume"] = all_data["KOSPI"]["Volume"]

merged = merged.fillna(method="ffill")

merged["KOSPI_Return"] = merged["KOSPI_Close"].pct_change()
merged["KOSPI_MA10"] = merged["KOSPI_Close"].rolling(10).mean()
merged["KOSPI_MA30"] = merged["KOSPI_Close"].rolling(30).mean()
merged["KOSPI_MA60"] = merged["KOSPI_Close"].rolling(60).mean()
merged["KOSPI_Volatility20"] = merged["KOSPI_Close"].rolling(20).std()

merged = merged.fillna(method="ffill")

merged_file = os.path.join(save_path, "KOSPI_merged_dataset_ffill.csv")
merged.to_csv(merged_file, encoding="utf-8-sig")

print("저장 완료.")
print(f"최종 통합 데이터셋: {merged_file}")

