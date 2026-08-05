import streamlit as st
import xarray as xr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter

st.set_page_config(layout="wide", page_title="CorrDiff 實驗評估 Dashboard")
st.title("🌧️ CorrDiff 降雨降尺度實驗評估 Dashboard")

# 1. 側邊欄參數調整區
st.sidebar.header("⚙️ 評估參數設定")
exp_selected = st.sidebar.multiselect(
    "選擇要比較的實驗 (Experiments)", 
    ["Exp_Baseline", "Exp_CorrDiff_v1", "Exp_CorrDiff_v2"], 
    default=["Exp_Baseline", "Exp_CorrDiff_v1"]
)

threshold = st.sidebar.slider("FSS 降雨門檻 (mm/hr)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
window_size = st.sidebar.slider("FSS 空間視窗大小 (Grid pixels)", min_value=1, max_value=21, value=5, step=2)

# 2. FSS 計算邏輯
def calculate_fss(obs, pred, threshold, window_size):
    obs_binary = (obs >= threshold).astype(float)
    pred_binary = (pred >= threshold).astype(float)
    
    # 空間遮罩平滑
    I_o = uniform_filter(obs_binary, size=window_size)
    I_p = uniform_filter(pred_binary, size=window_size)
    
    mse = np.mean((I_p - I_o) ** 2)
    ref_mse = np.mean(I_p ** 2) + np.mean(I_o ** 2)
    
    if ref_mse == 0:
        return 1.0
    return 1.0 - (mse / ref_mse)

# 3. 模擬計算結果 (實際應用時請替換為從 xarray 讀取 NetCDF)
st.subheader("📊 實驗指標對比")

# 繪製 RMSE Bar 圖
col1, col2 = st.columns(2)

with col1:
    st.markdown("### RMSE 比較")
    # 假設計算得到的 RMSE 數據
    rmse_data = {"Exp_Baseline": 4.52, "Exp_CorrDiff_v1": 3.21, "Exp_CorrDiff_v2": 2.85}
    filtered_rmse = {k: rmse_data[k] for k in exp_selected if k in rmse_data}
    
    fig_rmse = px.bar(
        x=list(filtered_rmse.keys()), 
        y=list(filtered_rmse.values()),
        labels={'x': '實驗組別', 'y': 'RMSE (mm/hr)'},
        title=f"各實驗整體 RMSE (越低越好)",
        color=list(filtered_rmse.keys())
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    st.markdown("### FSS 門檻分析")
    # 動態計算當前參數下的 FSS 數值
    fss_results = {}
    for exp in exp_selected:
        # 範例動態數據模擬，可連結真實模型輸出
        dummy_fss = np.clip(0.85 - (threshold * 0.01) + (window_size * 0.005) + np.random.uniform(-0.02, 0.02), 0, 1)
        fss_results[exp] = dummy_fss
        
    fig_fss = px.bar(
        x=list(fss_results.keys()),
        y=list(fss_results.values()),
        labels={'x': '實驗組別', 'y': 'FSS Score'},
        title=f"門檻 >= {threshold} mm/hr (視窗 {window_size}x{window_size}) 之下 FSS",
        color=list(fss_results.keys())
    )
    fig_fss.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_fss, use_container_width=True)
