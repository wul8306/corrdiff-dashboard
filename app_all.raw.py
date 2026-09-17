import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 頁面配置
# ==========================================
st.set_page_config(
    page_title="CorrDiff 降尺度實驗評估系統",
    page_icon="🌧️",
    layout="wide"
)

DB_PATH = "corrdiff_allmetrics.db"

# ==========================================
# 資料庫快取載入函式
# ==========================================
@st.cache_data
def load_experiment_metrics(db_path=DB_PATH):
    """讀取時間軸指標資料表 (experiment_metrics)"""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM experiment_metrics"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def get_grid_scatter_combos(db_path=DB_PATH):
    """取得格點 RMSE 散佈圖已預算的實驗組合清單"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT exp_x, exp_y FROM grid_rmse_scatter")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception:
        conn.close()
        return []

def load_grid_scatter_data(exp_x, exp_y, var_name="precipitation", db_path=DB_PATH):
    """讀取特定兩兩實驗組合的格點 RMSE 數據 (從 BLOB 還原)"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 正向查詢
    cursor.execute("""
    SELECT ratio_y_gt_x, x_values, y_values 
    FROM grid_rmse_scatter 
    WHERE exp_x=? AND exp_y=? AND var_name=?
    """, (exp_x, exp_y, var_name))
    row = cursor.fetchone()
    
    is_swapped = False
    if not row:
        # 反向查詢 (X/Y 對調)
        cursor.execute("""
        SELECT ratio_y_gt_x, x_values, y_values 
        FROM grid_rmse_scatter 
        WHERE exp_x=? AND exp_y=? AND var_name=?
        """, (exp_y, exp_x, var_name))
        row = cursor.fetchone()
        is_swapped = True

    conn.close()

    if not row:
        return None, None, None

    ratio_orig, x_blob, y_blob = row
    x_draw = np.frombuffer(x_blob, dtype=np.float32)
    y_draw = np.frombuffer(y_blob, dtype=np.float32)

    if is_swapped:
        x_draw, y_draw = y_draw, x_draw
        ratio_y_gt_x = 100.0 - ratio_orig
    else:
        ratio_y_gt_x = ratio_orig

    return ratio_y_gt_x, x_draw, y_draw


# ==========================================
# 側邊欄與頁籤選單
# ==========================================
st.sidebar.title("🌧️ CorrDiff 評估儀表板")
tab_selection = st.sidebar.radio(
    "請選擇分析模式：",
    ["📊 時間軸指標 (RMSE / FSS)", "🗺️ 不同實驗 RMSE 散佈圖"]
)

st.sidebar.markdown("---")
st.sidebar.caption("數據來源：SQLite 預計算資料庫 (`corrdiff_metrics.db`)")


# ==========================================
# TAB 1: 時間軸區域指標分析
# ==========================================
if tab_selection == "📊 時間軸指標 (RMSE / FSS)":
    st.title("📊 時間軸指標分析 (Prediction vs. Truth)")
    
    try:
        df_metrics = load_experiment_metrics()
    except Exception as e:
        st.error(f"❌ 無法讀取資料庫，請確認是否已執行預計算腳本：{e}")
        st.stop()

    if df_metrics.empty:
        st.warning("⚠️ 資料庫中尚無指標資料。")
        st.stop()

    # 側邊欄篩選條件
    all_exps = sorted(df_metrics['exp_name'].unique().tolist())
    selected_exps = st.sidebar.multiselect("選擇要比較的實驗：", options=all_exps, default=all_exps)

    if not selected_exps:
        st.info("請至少選擇一個實驗進行分析。")
        st.stop()

    df_filtered = df_metrics[df_metrics['exp_name'].isin(selected_exps)]

    st.subheader("1. 時間軸 RMSE 趨勢圖")
    # 對時間與實驗取平均 RMSE
    df_rmse = df_filtered.groupby(['exp_name', 'valid_time'])['rmse'].mean().reset_index()
    fig_rmse = px.line(
        df_rmse, x='valid_time', y='rmse', color='exp_name',
        title="Prediction vs. Truth 平均 RMSE 時間變化",
        labels={'valid_time': 'Valid Time', 'rmse': 'RMSE ', 'exp_name': '實驗名稱'}
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Fractional Skill Score (FSS) 分析")
    col1, col2 = st.columns(2)
    with col1:
        th_list = sorted(df_filtered['fss_threshold'].unique().tolist())
        selected_th = st.selectbox("選擇 FSS 門檻值 (Threshold, ):", options=th_list, index=0)
    with col2:
        w_list = sorted(df_filtered['fss_window_size'].unique().tolist())
        selected_w = st.selectbox("選擇 FSS 網格視窗大小 (Window Size):", options=w_list, index=2)

    df_fss = df_filtered[
        (df_filtered['fss_threshold'] == selected_th) & 
        (df_filtered['fss_window_size'] == selected_w)
    ].groupby(['exp_name', 'valid_time'])['fss_score'].mean().reset_index()

    fig_fss = px.line(
        df_fss, x='valid_time', y='fss_score', color='exp_name',
        title=f"FSS 分數時間變化 (門檻: {selected_th} , 視窗: {selected_w}x{selected_w})",
        labels={'valid_time': 'Valid Time', 'fss_score': 'FSS Score', 'exp_name': '實驗名稱'}
    )
    fig_fss.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig_fss, use_container_width=True)


# ==========================================
# TAB 2: 跨實驗格點 RMSE 散佈圖比較
# ==========================================
elif tab_selection == "🗺️ 跨實驗格點 RMSE 散佈圖":
    st.title("🗺️ 跨實驗格點 RMSE 比較 (Grid-level Scatter)")
    st.caption("比較兩個實驗在整個空間中每一個地理格點的 `pred_vs_truth` 平均 RMSE。")

    combos = get_grid_scatter_combos()

    if not combos:
        st.error("❌ 資料庫中找不到預算好的格點 RMSE 數據，請確認是否已執行 `process_all.py`。")
        st.stop()

    all_grid_exps = sorted(list(set([r[0] for r in combos] + [r[1] for r in combos])))

    col_x, col_y = st.columns(2)
    with col_x:
        exp_x = st.selectbox("選擇 X 軸實驗 (Baseline)", options=all_grid_exps, index=0)
    with col_y:
        default_y_idx = 1 if len(all_grid_exps) > 1 else 0
        exp_y = st.selectbox("選擇 Y 軸實驗 (Comparison)", options=all_grid_exps, index=default_y_idx)

    if exp_x == exp_y:
        st.warning("⚠️ 請選擇兩個不同的實驗進行比較。")
    else:
        ratio_y_gt_x, x_draw, y_draw = load_grid_scatter_data(exp_x, exp_y, var_name="precipitation")

        if x_draw is None:
            st.error(f"⚠️ 資料庫中找不到 `{exp_x}` 與 `{exp_y}` 的對比數據。")
        else:
            ratio_x_gt_y = 100.0 - ratio_y_gt_x

            # 繪製 Scatter Plot
            fig_scatter = px.scatter(
                x=x_draw,
                y=y_draw,
                opacity=0.35,
                labels={
                    'x': f"{exp_x} 格點 RMSE ",
                    'y': f"{exp_y} 格點 RMSE "
                },
                title=f"【Pred vs. Truth 格點 RMSE 對比】{exp_x} (X軸) vs. {exp_y} (Y軸)"
            )

            # 計算邊界與加入 1:1 紅色對角參考線 (y = x)
            max_val = max(float(x_draw.max()), float(y_draw.max())) * 1.05
            fig_scatter.add_shape(
                type="line",
                x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="Red", width=2, dash="dash")
            )

            fig_scatter.update_xaxes(range=[0, max_val])
            fig_scatter.update_yaxes(range=[0, max_val])

            # 在左上角加入比例文字標註 (Annotation)
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

            # 允許客製化下載檔名與高高清晰度 PNG
            st.plotly_chart(
                fig_scatter, 
                use_container_width=True,
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{exp_x}_vs_{exp_y}_grid_rmse',
                        'height': 600,
                        'width': 800,
                        'scale': 2
                    }
                }
            )

            st.caption(
                "💡 **圖表解讀說明**：\n"
                f"- **Y > X 比例 ({ratio_y_gt_x:.1f}%)**：代表有此比例的地理格點中，**{exp_y} (Y軸)** 的誤差高於 **{exp_x} (X軸)**。\n"
                f"- **落在紅色虛線 ($y = x$) 下方**：代表 **{exp_y} (Y軸)** 在該格點降尺度表現較好。\n"
                f"- **落在紅色虛線上方**：代表 **{exp_x} (X軸)** 在該格點降尺度表現較好。"
            )
