import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="CorrDiff 實驗評估 Dashboard")
st.title("🌧️ CorrDiff 降雨降尺度實驗評估 Dashboard")

DB_PATH = "corrdiff_metrics.db"

# 使用快取載入 SQLite 資料
@st.cache_data
def load_metrics_data(db_path):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT exp_name, file_name, valid_time, comparison_type, variable_name, 
           rmse, fss_threshold, fss_window_size, fss_score 
    FROM experiment_metrics
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 可讀性優化：將 comparison_type 轉換為中文易讀標籤
    type_map = {
        'pred_vs_truth': 'Prediction vs Truth (模型降尺度)',
        'input_vs_truth': 'Input vs Truth (低解析輸入)'
    }
    df['comparison_label'] = df['comparison_type'].map(type_map).fillna(df['comparison_type'])
    return df

try:
    df = load_metrics_data(DB_PATH)
except Exception as e:
    st.error(f"⚠️ 無法讀取資料庫 `{DB_PATH}`。請確認預處理腳本已順利執行完成。錯誤訊息：{e}")
    st.stop()

# 側邊欄控制區
st.sidebar.header("⚙️ 評估參數設定")

# 動態從資料庫讀取已計算的選單
available_exps = sorted(df['exp_name'].unique().tolist())
available_thresholds = sorted(df['fss_threshold'].unique().tolist())
available_windows = sorted(df['fss_window_size'].unique().tolist())

exp_selected = st.sidebar.multiselect(
    "選擇要對比的實驗 (Experiments)",
    options=available_exps,
    default=available_exps
)

# 使用 select_slider 鎖定資料庫中實際有預算的數值
threshold = st.sidebar.select_slider(
    "FSS 降雨門檻 (mm/hr)",
    options=available_thresholds,
    value=available_thresholds[min(3, len(available_thresholds) - 1)]
)

window_size = st.sidebar.select_slider(
    "FSS 空間視窗大小 (Grid pixels)",
    options=available_windows,
    value=available_windows[min(1, len(available_windows) - 1)]
)

# 篩選選定條件的數據
filtered_df = df[
    (df['exp_name'].isin(exp_selected)) &
    (df['fss_threshold'] == threshold) &
    (df['fss_window_size'] == window_size)
]

st.subheader("📊 實驗指標與基準對比 (Prediction vs. Input vs. Truth)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**RMSE 比較 (越低越好)**")
    rmse_summary = filtered_df.groupby(['exp_name', 'comparison_label'], as_index=False)['rmse'].mean()
    
    fig_rmse = px.bar(
        rmse_summary,
        x='exp_name',
        y='rmse',
        color='comparison_label',
        barmode='group',  # 使用分組長條圖
        labels={'exp_name': '實驗組別', 'rmse': '平均 RMSE (mm/hr)', 'comparison_label': '對比對象'},
        title="Prediction vs. Input 平均 RMSE"
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    st.markdown("**FSS 門檻分析 (越高越好)**")
    fss_summary = filtered_df.groupby(['exp_name', 'comparison_label'], as_index=False)['fss_score'].mean()
    
    fig_fss = px.bar(
        fss_summary,
        x='exp_name',
        y='fss_score',
        color='comparison_label',
        barmode='group',  # 使用分組長條圖
        labels={'exp_name': '實驗組別', 'fss_score': 'FSS Score', 'comparison_label': '對比對象'},
        title=f"門檻 ≥ {threshold} mm/hr (視窗 {window_size}x{window_size}) 之 FSS"
    )
    fig_fss.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_fss, use_container_width=True)

# 延伸分析：跨門檻之 FSS 趨勢圖 (FSS Spectrum)
st.markdown("---")
st.subheader("📈 跨降雨門檻之 FSS 技能分數趨勢 (FSS Spectrum)")

spectrum_df = df[
    (df['exp_name'].isin(exp_selected)) &
    (df['fss_window_size'] == window_size)
].groupby(['exp_name', 'comparison_label', 'fss_threshold'], as_index=False)['fss_score'].mean()

fig_spectrum = px.line(
    spectrum_df,
    x='fss_threshold',
    y='fss_score',
    color='exp_name',
    line_dash='comparison_label',  # 實線表 Prediction，虛線表 Input
    markers=True,
    labels={'fss_threshold': '降雨門檻 (mm/hr)', 'fss_score': 'FSS Score', 'exp_name': '實驗組別', 'comparison_label': '對比對象'},
    title=f"視窗 {window_size}x{window_size} 下，FSS 隨降雨強度增加之衰減曲線"
)
fig_spectrum.update_yaxes(range=[0, 1])
st.plotly_chart(fig_spectrum, use_container_width=True)
