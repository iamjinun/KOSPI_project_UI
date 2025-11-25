import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, date
import math
import os
import holidays 

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out, attn_weights

class CNNLSTM(nn.Module):
    def __init__(self, num_features=10, cnn_channels=64, lstm_hidden=256, lstm_layers=2, fc_hidden=64, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(cnn_channels)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0, bidirectional=True)
        self.attention = Attention(lstm_hidden * 2)
        self.fc1 = nn.Linear(lstm_hidden * 2, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        out = attn_out.mean(dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        return out.squeeze(-1)
    
@st.cache_resource
def load_trained_model(path, device):
    if not os.path.exists(path):
        return None
    model = CNNLSTM(num_features=10)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except Exception as e:
        st.error(f"모델 로드 에러: {e}")
        return None
    model.to(device)
    model.eval()
    return model

def get_market_data(end_date_str, seq_len=50):
    today = pd.Timestamp.now().normalize()
    start_date = today - timedelta(days=200)
    
    tickers = {
        "KOSPI": "^KS11", "SP500": "^GSPC", "USD_KRW": "KRW=X",
        "WTI_OIL": "CL=F", "GOLD": "GC=F"
    }
    
    all_data = {}
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
        if data.empty:
            return None, None
        all_data[name] = data

    full_dates = pd.date_range(start=start_date, end=today, freq="B")
    
    kospi = all_data["KOSPI"][["Open", "High", "Low", "Close"]].copy()
    if isinstance(kospi.columns, pd.MultiIndex):
        kospi.columns = kospi.columns.get_level_values(0)

    merged = kospi.rename(columns={
        "Open": "KOSPI_Open", "High": "KOSPI_High", "Low": "KOSPI_Low", "Close": "KOSPI_Close"
    })
    
    merged = merged.reindex(full_dates).ffill()
    
    for key, val in all_data.items():
        if key == "KOSPI": continue
        try:
            temp = val["Close"].copy()
            if isinstance(temp, pd.DataFrame): 
                temp = temp.iloc[:, 0]
            merged[f"{key}_Close"] = temp
        except:
            merged[f"{key}_Close"] = 0.0 
            
    merged = merged.reindex(full_dates).ffill()

    merged["KOSPI_Return"] = merged["KOSPI_Close"].pct_change()
    merged["KOSPI_Volatility20"] = merged["KOSPI_Close"].rolling(20).std()
    
    merged = merged.dropna()
    return merged, tickers

def is_market_open(target_date, kr_holidays):
    if target_date.weekday() >= 5:
        return False
    
    if target_date in kr_holidays:
        return False

    if target_date.month == 12 and target_date.day == 31:
        return False
        
    return True

def predict_gap_and_future(model, df, seq_len, device, target_start_date, predict_days=5):
    feature_cols = [
        "KOSPI_Open", "KOSPI_High", "KOSPI_Low", "KOSPI_Close",
        "SP500_Close", "USD_KRW_Close", "WTI_OIL_Close", "GOLD_Close",
        "KOSPI_Return", "KOSPI_Volatility20"
    ]
    
    scaler_x = MinMaxScaler()
    data_values = df[feature_cols].values
    scaler_x.fit(data_values)
    
    close_diffs = df["KOSPI_Close"].diff().dropna().values.reshape(-1, 1)
    scaler_y = MinMaxScaler()
    scaler_y.fit(close_diffs)

    current_seq = data_values[-seq_len:] 
    last_real_date = df.index[-1]
    last_close = df["KOSPI_Close"].iloc[-1]
    
    target_start_dt = pd.to_datetime(target_start_date)
    
    all_pred_dates = []
    all_pred_prices = []
    
    current_date = last_real_date
    max_steps = 60 
    steps = 0
    collected_predictions = 0

    kr_holidays = holidays.KR()

    while steps < max_steps:

        current_date += timedelta(days=1)

        if not is_market_open(current_date, kr_holidays):
            continue
            
        seq_scaled = scaler_x.transform(current_seq)
        input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_delta_scaled = model(input_tensor).item()
        
        pred_delta = scaler_y.inverse_transform([[pred_delta_scaled]])[0][0]
        next_close = last_close + pred_delta
        
        all_pred_dates.append(current_date)
        all_pred_prices.append(next_close)
        
        if current_date >= target_start_dt:
            collected_predictions += 1
        
        if collected_predictions >= predict_days:
            break
        
        next_row = current_seq[-1].copy()
        next_row[0] = next_close 
        next_row[1] = next_close * 1.002
        next_row[2] = next_close * 0.998
        next_row[3] = next_close
        next_row[8] = (next_close - last_close) / last_close
        next_row[9] = next_row[9]
        
        current_seq = np.vstack([current_seq[1:], next_row])
        last_close = next_close
        steps += 1

    return all_pred_dates, all_pred_prices, last_real_date

def main():
    st.set_page_config(page_title="KOSPI Prediction", layout="wide")
    
    st.markdown("""
        <style>
        .main { background-color: #FFFFFF; }
        div.stButton > button {
            width: 100%;
            background-color: #000000;
            color: #FFFFFF;
            border-radius: 5px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("KOSPI Prediction AI")
    st.markdown("### 미래 주가 예측 시뮬레이션")
    st.divider()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_cnn_lstm_kospi.pth"
    model = load_trained_model(model_path, device)

    if model is None:
        st.error(f"모델 파일({model_path})이 없습니다.")
        st.stop()

    with st.spinner("데이터 동기화 중..."):
        df_init, _ = get_market_data(None)
        if df_init is None:
            st.error("데이터 로드 실패")
            st.stop()
        last_real_date = df_init.index[-1]

    with st.sidebar:
        st.header("설정 (Settings)")
        st.write(f"**데이터 기준일:** {last_real_date.strftime('%Y-%m-%d (%a)')}")
        
        min_date = last_real_date + timedelta(days=1)
        max_date_limit = last_real_date + timedelta(days=30)
        
        selected_date = st.date_input(
            "예측 시작일 선택", 
            min_value=min_date,
            max_value=max_date_limit,
            value=min_date
        )

        target_start = pd.to_datetime(selected_date)
        kr_holidays = holidays.KR()
        
        if target_start <= last_real_date:
            st.error(f"과거 날짜입니다. {last_real_date.strftime('%Y-%m-%d')} 이후를 선택해주세요.")
            valid_date = False
        elif target_start > max_date_limit:
            st.error(f"예측 불가능한 범위입니다.\n데이터 기준일로부터 1개월 이내({max_date_limit.strftime('%Y-%m-%d')})까지만 가능합니다.")
            valid_date = False
        else:
            valid_date = True

            temp_date = target_start
            count = 1
            while count < 5: 
                temp_date += timedelta(days=1)
                if is_market_open(temp_date, kr_holidays):
                    count += 1
            target_end = temp_date
            
            start_str = target_start.strftime('%Y-%m-%d (%a)')
            end_str = target_end.strftime('%Y-%m-%d (%a)')
            
            st.success(f"**예측 구간**\n\n{start_str} ~ {end_str}")
            
            gap_days = (target_start - last_real_date).days - 1
            if gap_days < 0: gap_days = 0
            
            if gap_days > 0:
                st.info(f"선택한 날짜까지의 공백({gap_days - 1}일)을 먼저 예측한 후 결과를 생성합니다.")

    if valid_date:
        if st.button("예측 실행 (Start Prediction)"):
            with st.spinner('예측 수행 중...'):
                seq_len = 50
                df, _ = get_market_data(None, seq_len)
                
                pred_dates, pred_prices, _ = predict_gap_and_future(
                    model, df, seq_len, device, selected_date
                )
                
                fig = go.Figure()

                display_df = df.iloc[-60:]
                fig.add_trace(go.Scatter(
                    x=display_df.index, 
                    y=display_df["KOSPI_Close"],
                    mode='lines',
                    name='실제 데이터 (History)',
                    line=dict(color='black', width=2)
                ))

                connect_x = [display_df.index[-1]] + pred_dates
                connect_y = [display_df["KOSPI_Close"].iloc[-1]] + pred_prices

                fig.add_trace(go.Scatter(
                    x=connect_x, 
                    y=connect_y,
                    mode='lines+markers',
                    name='AI 예측 경로',
                    line=dict(color='#FF4136', width=2, dash='dot'),
                    marker=dict(size=6)
                ))

                title_date_str = pd.to_datetime(selected_date).strftime('%Y-%m-%d (%a)')
                
                fig.update_layout(
                    title=f"KOSPI 주가 예측 시뮬레이션 ({title_date_str} 기준)",
                    xaxis_title="날짜",
                    yaxis_title="주가 (KRW)",
                    template="plotly_white",
                    hovermode="x unified",
                    autosize=True,
                    xaxis=dict(
                        tickformat="%Y-%m-%d",
                        dtick="D1"
                    )
                )

                st.plotly_chart(fig, config={'responsive': True})

                st.subheader(f"{title_date_str} 부터 5일간 예측 결과")
                
                final_dates = []
                final_prices = []
                
                target_ts = pd.to_datetime(selected_date)
                
                for d, p in zip(pred_dates, pred_prices):
                    if d >= target_ts:
                        final_dates.append(d)
                        final_prices.append(p)
                        if len(final_dates) == 5: break 

                res_df = pd.DataFrame({
                    "날짜": [d.strftime("%Y-%m-%d (%a)") for d in final_dates],
                    "예측 주가": [f"{p:,.0f} 원" for p in final_prices]
                })
                
                st.table(res_df)

    else:
        st.warning("왼쪽 사이드바에서 날짜를 선택해주세요.")

if __name__ == "__main__":
    main()