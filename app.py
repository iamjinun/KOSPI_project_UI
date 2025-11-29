import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import os
import holidays
from model import CNNLSTM
import json

CSV_PATH = "KOSPI_merged_dataset_ffill.csv"
MODEL_PATH = "best_cnn_lstm_kospi.pth"
SEQ_LEN = 30

FEATURE_COLS = [
    "KOSPI_Open", "KOSPI_High", "KOSPI_Low", "KOSPI_Close",
    "SP500_Close", "USD_KRW_Close", "WTI_OIL_Close", "GOLD_Close",
    "KOSPI_Return", "KOSPI_Volatility20",
    "KOSPI_Volume", "KOSPI_MA10", "KOSPI_MA30", "KOSPI_MA60"
]

@st.cache_resource
def load_trained_model(path, device):
    if not os.path.exists(path):
        return None

    model = CNNLSTM(num_features=len(FEATURE_COLS))
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except Exception as e:
        st.error(f"모델 로드 에러: {e}")
        return None

    model.to(device)
    model.eval()
    return model

@st.cache_data
def get_fixed_market_data(csv_path, seq_len):
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)

        date_col = next((col for col in ['Date', 'Unnamed: 0'] if col in df.columns), df.columns[0])
        
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
        
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col).sort_index()

        if 'KOSPI_Close' in df.columns:
             df['KOSPI_Close'] = pd.to_numeric(df['KOSPI_Close'], errors='coerce')
        
        df = df[FEATURE_COLS].copy()
        df = df.dropna()

        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None


def is_market_open(target_date, kr_holidays):
    if target_date.weekday() >= 5: return False
    if target_date in kr_holidays: return False
    if target_date.month == 12 and target_date.day == 31: return False
    return True

def get_next_business_day(start_date, kr_holidays):
    current = start_date
    while not is_market_open(current, kr_holidays):
        current += timedelta(days=1)
    return current

def predict_simulation(model, df, seq_len, device, target_start_date):
    scaler_x = MinMaxScaler()
    scaler_x.fit(df[FEATURE_COLS].values)

    close_diffs = df["KOSPI_Close"].diff().dropna().values.reshape(-1, 1)
    scaler_y = MinMaxScaler()
    scaler_y.fit(close_diffs)

    current_seq = df[FEATURE_COLS].values[-seq_len:]

    last_real_date = df.index[-1]
    running_close = df["KOSPI_Close"].iloc[-1]

    target_start_dt = pd.to_datetime(target_start_date)
    kr_holidays = holidays.KR()

    real_target_start = get_next_business_day(target_start_dt, kr_holidays)

    target_end_date = real_target_start + timedelta(days=15)

    all_pred_dates = []
    all_pred_prices = []

    current_date = last_real_date

    max_loops = 50

    for _ in range(max_loops):
        seq_scaled = scaler_x.transform(current_seq)
        input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_deltas_scaled = model(input_tensor).cpu().numpy().flatten()

        pred_deltas = scaler_y.inverse_transform(pred_deltas_scaled.reshape(1, -1)).flatten()


        temp_prices = []
        temp_dates = []
        base = running_close

        for i in range(5):
            while True:
                current_date += timedelta(days=1)
                if is_market_open(current_date, kr_holidays):
                    break

            next_price = base + pred_deltas[i]
            if next_price < 1: next_price = 1
            temp_dates.append(current_date)
            temp_prices.append(next_price)
            base = next_price

        all_pred_dates.extend(temp_dates)
        all_pred_prices.extend(temp_prices)
        running_close = temp_prices[-1]

        if len(all_pred_dates) >= 5 and all_pred_dates[-1] >= target_end_date:
            break
        if len(all_pred_dates) >= 50:
             break

        new_rows = []
        temp_running_close_for_feat = current_seq[-1, 3]

        for i in range(5):
            p_close = temp_prices[i]
            prev_close = temp_running_close_for_feat
            temp_running_close_for_feat = p_close

            row = np.zeros(len(FEATURE_COLS))
            row[0] = p_close
            row[1] = p_close * 1.002
            row[2] = p_close * 0.998
            row[3] = p_close
            row[4] = current_seq[-1, 4]
            row[5] = current_seq[-1, 5]
            row[6] = current_seq[-1, 6]
            row[7] = current_seq[-1, 7]
            row[8] = (p_close - prev_close) / prev_close if prev_close != 0 else 0
            row[9] = current_seq[-1, 9]
            row[10] = current_seq[-1, 10]
            alpha = 0.1
            row[11] = current_seq[-1, 11] * (1-alpha) + p_close * alpha
            row[12] = current_seq[-1, 12] * (1-alpha) + p_close * alpha
            row[13] = current_seq[-1, 13] * (1-alpha) + p_close * alpha

            new_rows.append(row)

        new_rows = np.array(new_rows)
        current_seq = np.vstack([current_seq[5:], new_rows])


    return all_pred_dates, all_pred_prices, last_real_date, real_target_start

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

    st.title("KOSPI 주가 예측 시뮬레이션")
    st.divider()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(MODEL_PATH, device)

    if model is None:
        st.error(f"모델 파일({MODEL_PATH})이 없습니다.")
        st.stop()

    with st.spinner("저장된 시장 데이터 로드 중..."):
        if 'market_data' not in st.session_state:
            df_init = get_fixed_market_data(CSV_PATH, SEQ_LEN)
            st.session_state['market_data'] = df_init
        else:
            df_init = st.session_state['market_data']

        if df_init is None or df_init.empty:
            st.error(f"저장된 데이터 파일({CSV_PATH}) 로드 실패")
            st.stop()

        last_real_date = df_init.index[-1]
        st.info(f"데이터 파일 기준일: {last_real_date.strftime('%Y-%m-%d (%a)')}")

    with st.sidebar:
        st.header("설정 (Settings)")
        st.write(f"**데이터 기준일:** {last_real_date.strftime('%Y-%m-%d (%a)')}")

        # 오늘 날짜를 기준으로 설정
        today = pd.Timestamp.now().date()
        min_start_date = today + timedelta(days=1)
        # 예측 가능 날짜를 앱 실행일 기준 약 한 달 후(31일)까지로 설정
        max_start_date = today + timedelta(days=31)
        
        # min_value를 오늘 날짜 + 1일로 설정하여 이전 날짜 선택 불가능하게 함
        selected_date = st.date_input(
            "예측 시작일 선택",
            min_value=min_start_date,
            max_value=max_start_date,
            value=min_start_date
        )

        target_start = pd.to_datetime(selected_date)
        kr_holidays = holidays.KR()
        real_start_date = get_next_business_day(target_start, kr_holidays)

        gap_days = (real_start_date - last_real_date).days
        if gap_days > 1:
            st.info(f"데이터 기준일({last_real_date.date()})부터 목표일까지의 공백은 AI가 예측하여 채웁니다.")

        temp_date = real_start_date
        calc_end = temp_date
        days_found = 1
        while days_found < 5:
            calc_end += timedelta(days=1)
            if is_market_open(calc_end, kr_holidays):
                days_found += 1

        start_str = real_start_date.strftime('%Y-%m-%d (%a)')
        end_str = calc_end.strftime('%Y-%m-%d (%a)')

        st.success(f"**실제 예측 구간**\n\n{start_str} ~ {end_str}")

    if 'pred_results' not in st.session_state:
        st.session_state['pred_results'] = None

    if 'last_selected_date' not in st.session_state:
        st.session_state['last_selected_date'] = selected_date

    if st.session_state['last_selected_date'] != selected_date:
        st.session_state['pred_results'] = None
        st.session_state['last_selected_date'] = selected_date

    if st.button("예측 실행 (Start Prediction)"):
        with st.spinner('AI 분석 및 예측 수행 중...'):
            all_dates, all_prices, _, real_start_dt = predict_simulation(
                model, df_init, SEQ_LEN, device, selected_date
            )
            st.session_state['pred_results'] = {
                'all_dates': all_dates,
                'all_prices': all_prices,
                'real_start_dt': real_start_dt
            }
    
    if st.session_state['pred_results'] is not None:
        results = st.session_state['pred_results']
        all_dates = results['all_dates']
        all_prices = results['all_prices']
        real_start_dt = results['real_start_dt']

        final_dates_tbl = []
        final_prices_tbl = []

        for d, p in zip(all_dates, all_prices):
            if d >= real_start_dt:
                final_dates_tbl.append(d)
                final_prices_tbl.append(p)
                if len(final_dates_tbl) == 5:
                    break

        if final_dates_tbl:
            plot_end_date = final_dates_tbl[-1]
        else:
            plot_end_date = all_dates[-1]

        plot_dates = []
        plot_prices = []

        for d, p in zip(all_dates, all_prices):
            plot_dates.append(d)
            plot_prices.append(p)
            if d >= plot_end_date:
                break

        fig = go.Figure()

        display_df = df_init.iloc[-60:]
        fig.add_trace(go.Scatter(
            x=display_df.index,
            y=display_df["KOSPI_Close"],
            mode='lines',
            name='실제 데이터 (History)',
            line=dict(color='black', width=2)
        ))

        connect_x = [display_df.index[-1]] + plot_dates
        connect_y = [display_df["KOSPI_Close"].iloc[-1]] + plot_prices

        fig.add_trace(go.Scatter(
            x=connect_x,
            y=connect_y,
            mode='lines+markers',
            name='AI 예측 경로',
            line=dict(color='#FF4136', width=2, dash='dot'),
            marker=dict(size=5)
        ))

        title_date_str = real_start_dt.strftime('%Y-%m-%d')
        fig.update_layout(
            title=f"KOSPI 주가 예측 시뮬레이션 (Target: {title_date_str})",
            xaxis_title="날짜",
            yaxis_title="주가 (KRW)",
            template="plotly_white",
            hovermode="x unified",
            autosize=True
        )
        st.plotly_chart(fig, config={'responsive': True})

        st.subheader(f"{title_date_str} 부터 5일간 예측 결과")

        res_df = pd.DataFrame({
            "날짜": [d.strftime("%Y-%m-%d (%a)") for d in final_dates_tbl],
            "예측 주가": [f"{p:,.0f} 원" for p in final_prices_tbl]
        })

        if not res_df.empty:
            st.table(res_df)
        else:
            st.warning("해당 구간의 예측 결과를 생성하지 못했습니다.")

if __name__ == "__main__":
    main()