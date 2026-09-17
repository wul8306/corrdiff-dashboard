import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        barmode='group',
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
        barmode='group',
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
    line_dash='comparison_label',
    markers=True,
    labels={'fss_threshold': '降雨門檻 (mm/hr)', 'fss_score': 'FSS Score', 'exp_name': '實驗組別', 'comparison_label': '對比對象'},
    title=f"視窗 {window_size}x{window_size} 下，FSS 隨降雨強度增加之衰減曲線"
)
fig_spectrum.update_yaxes(range=[0, 1])
st.plotly_chart(fig_spectrum, use_container_width=True)


st.markdown("---")
st.subheader("🗺️ 跨實驗格點 RMSE 比較散佈圖 (預算資料庫秒載入)")

DB_PATH = "corrdiff_rmse_metrics.db"

# 1. 取得 DB 中已儲存的預算組合
@st.cache_data
def get_stored_scatter_combos(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT exp_x, exp_y FROM grid_rmse_scatter")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except:
        conn.close()
        return []

stored_combos = get_stored_scatter_combos(DB_PATH)

if not stored_combos:
    st.info("💡 資料庫中尚無預算的格點 RMSE 數據，請先執行預算腳本 `process_grid_rmse.py`。")
else:
    all_exps = sorted(list(set([r[0] for r in stored_combos] + [r[1] for r in stored_combos])))

    col_x, col_y = st.columns(2)
    with col_x:
        exp_x = st.selectbox("選擇 X 軸實驗", options=all_exps, index=0)
    with col_y:
        default_y_idx = 1 if len(all_exps) > 1 else 0
        exp_y = st.selectbox("選擇 Y 軸實驗", options=all_exps, index=default_y_idx)

    # 處理正反向 (X vs Y 或 Y vs X)
    is_swapped = False
    query_x, query_y = exp_x, exp_y

    if (exp_x, exp_y) not in stored_combos and (exp_y, exp_x) in stored_combos:
        query_x, query_y = exp_y, exp_x
        is_swapped = True

    if exp_x == exp_y:
        st.warning("⚠️ 請選擇兩個不同的實驗。")
    elif (query_x, query_y) not in stored_combos:
        st.error(f"⚠️ 資料庫中找不到 `{exp_x}` 與 `{exp_y}` 的配對數據。")
    else:
        # 從 DB 撈取 BLOB 數據
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT ratio_y_gt_x, x_values, y_values
        FROM grid_rmse_scatter
        WHERE exp_x=? AND exp_y=? AND var_name='precipitation'
        """, (query_x, query_y))
        row = cursor.fetchone()
        conn.close()

        ratio_orig, x_blob, y_blob = row

        # 將 BLOB 還原為 Numpy 陣列
        x_draw = np.frombuffer(x_blob, dtype=np.float32)
        y_draw = np.frombuffer(y_blob, dtype=np.float32)

        # 若使用者選取的 X/Y 軸順序與 DB 紀錄相反，進行反轉處理
        if is_swapped:
            x_draw, y_draw = y_draw, x_draw
            ratio_y_gt_x = 100.0 - ratio_orig
        else:
            ratio_y_gt_x = ratio_orig

        ratio_x_gt_y = 100.0 - ratio_y_gt_x

        # 繪製 Scatter Plot
        fig_scatter = px.scatter(
            x=x_draw,
            y=y_draw,
            opacity=0.35,
            labels={
                'x': f"{exp_x} 格點 RMSE (mm/hr)",
                'y': f"{exp_y} 格點 RMSE (mm/hr)"
            },
            title=f"【Exp1 vs. Exp2 格點 RMSE 對比】{exp_x} (X軸) vs. {exp_y} (Y軸)"
        )

        max_val = max(x_draw.max(), y_draw.max()) * 1.05
        fig_scatter.add_shape(
            type="line",
            x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="Red", width=2, dash="dash")
        )

        fig_scatter.update_xaxes(range=[0, max_val])
        fig_scatter.update_yaxes(range=[0, max_val])

        # 左上角標註
        annotation_text = (
            f"<b>Y > X (Y軸較差) 比例: {ratio_y_gt_x:.1f}%</b><br>"
            f"X > Y (X軸較差) 比例: {ratio_x_gt_y:.1f}%"
        )

        fig_scatter.add_annotation(
            xref="paper", yref="paper",
            x=0.03, y=0.95,
            text=annotation_text,
            showarrow=False,
            align="left",
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=6
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
