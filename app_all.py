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
def load_all_time_metrics(db_path=DB_PATH):
    """讀取全時間平均指標資料表 (experiment_metrics)"""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT exp_name, file_name, comparison_type, variable_name, 
           rmse, fss_threshold, fss_window_size, fss_score
    FROM experiment_metrics
    WHERE comparison_type = 'pred_vs_truth'
    """
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
    ["📊 全時間平均指標 (RMSE & FSS)", "🗺️ 跨實驗格點 RMSE 散佈圖"]
)

st.sidebar.markdown("---")
st.sidebar.caption("數據來源：SQLite 預計算資料庫 (`corrdiff_metrics.db`)")


# ==========================================
# TAB 1: 全時間平均指標分析 (RMSE & FSS)
# ==========================================
if tab_selection == "📊 全時間平均指標 (RMSE & FSS)":
    st.title("📊 各實驗全時間平均指標 (Prediction vs. Truth)")
    
    try:
        df_metrics = load_all_time_metrics()
    except Exception as e:
        st.error(f"❌ 無法讀取資料庫，請確認是否已執行預計算腳本：{e}")
        st.stop()

    if df_metrics.empty:
        st.warning("⚠️ 資料庫中尚無指標資料。")
        st.stop()

    # 側邊欄實驗篩選
    all_exps = sorted(df_metrics['exp_name'].unique().tolist())
    selected_exps = st.sidebar.multiselect("選擇要比較的實驗：", options=all_exps, default=all_exps)

    if not selected_exps:
        st.info("請至少選擇一個實驗進行分析。")
        st.stop()

    df_filtered = df_metrics[df_metrics['exp_name'].isin(selected_exps)]

    # ------------------------------------------
    # 1. 總平均 RMSE 比較區
    # ------------------------------------------
    st.subheader("1. 跨實驗總平均 RMSE 對比 (越低越好)")
    
    # 計算各實驗的平均 RMSE (去除重複的 FSS 組合影響)
    df_rmse_avg = df_filtered.groupby('exp_name', as_index=False)['rmse'].mean().sort_values('rmse')

    # KPI 卡片
    cols = st.columns(len(df_rmse_avg))
    best_rmse = df_rmse_avg['rmse'].min()

    for idx, row in df_rmse_avg.reset_index(drop=True).iterrows():
        exp_name = row['exp_name']
        rmse_val = row['rmse']
        with cols[idx]:
            if rmse_val == best_rmse:
                st.metric(label=f"🏆 {exp_name} (最佳)", value=f"{rmse_val:.4f} mm/hr")
            else:
                diff = rmse_val - best_rmse
                st.metric(label=exp_name, value=f"{rmse_val:.4f} mm/hr", delta=f"+{diff:.4f}", delta_color="inverse")

    # RMSE 長條圖
    fig_rmse = px.bar(
        df_rmse_avg,
        x='exp_name',
        y='rmse',
        color='exp_name',
        text_auto='.4f',
        title="Prediction vs. Truth 全時間平均 RMSE (mm/hr)",
        labels={'exp_name': '實驗名稱', 'rmse': '平均 RMSE (mm/hr)'}
    )
    fig_rmse.update_traces(textposition='outside')
    fig_rmse.update_layout(showlegend=False, yaxis_range=[0, df_rmse_avg['rmse'].max() * 1.25])
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------
    # 2. 全時間平均 FSS 分數比較區
    # ------------------------------------------
    st.subheader("2. Fractional Skill Score (FSS) 平均表現 (越高越好)")
    
    fss_tab1, fss_tab2 = st.tabs(["🎯 特定門檻/視窗長條圖", "🔥 FSS 矩陣熱力圖 (Heatmap)"])

    with fss_tab1:
        col1, col2 = st.columns(2)
        with col1:
            th_list = sorted(df_filtered['fss_threshold'].unique().tolist())
            selected_th = st.selectbox("選擇降雨門檻值 (Threshold, mm/hr):", options=th_list, index=2)
        with col2:
            w_list = sorted(df_filtered['fss_window_size'].unique().tolist())
            selected_w = st.selectbox("選擇空間視窗大小 (Window Size):", options=w_list, index=2)

        df_fss_sub = df_filtered[
            (df_filtered['fss_threshold'] == selected_th) & 
            (df_filtered['fss_window_size'] == selected_w)
        ].groupby('exp_name', as_index=False)['fss_score'].mean().sort_values('fss_score', ascending=False)

        fig_fss_bar = px.bar(
            df_fss_sub,
            x='exp_name',
            y='fss_score',
            color='exp_name',
            text_auto='.3f',
            title=f"平均 FSS 分數 (門檻: {selected_th} mm/hr, 視窗: {selected_w}x{selected_w})",
            labels={'exp_name': '實驗名稱', 'fss_score': 'FSS Score'}
        )
        fig_fss_bar.update_traces(textposition='outside')
        fig_fss_bar.update_layout(showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_fss_bar, use_container_width=True)

    with fss_tab2:
        st.markdown("##### 選擇實驗觀看「降雨門檻 vs 視窗大小」的全域 FSS 分佈矩陣：")
        exp_for_hm = st.selectbox("選擇實驗：", options=selected_exps, key="hm_exp")
        
        df_hm = df_filtered[df_filtered['exp_name'] == exp_for_hm].groupby(
            ['fss_threshold', 'fss_window_size'], as_index=False
        )['fss_score'].mean()

        pivot_fss = df_hm.pivot(index='fss_window_size', columns='fss_threshold', values='fss_score')

        fig_hm = px.imshow(
            pivot_fss,
            text_auto='.3f',
            color_continuous_scale='Viridis',
            zmin=0, zmax=1,
            labels=dict(x="視窗大小 (Window Size)", y="降雨門檻 (Threshold mm/hr)", color="FSS"),
            title=f"【{exp_for_hm}】不同門檻與視窗下之平均 FSS 熱力圖"
        )
        st.plotly_chart(fig_hm, use_container_width=True)


# ==========================================
# TAB 2: 跨實驗格點 RMSE 散佈圖比較
# ==========================================
elif tab_selection == "🗺️ 跨實驗格點 RMSE 散佈圖":
    st.title("🗺️ 跨實驗格點 RMSE 比較 (Grid-level Scatter)")
    st.caption("比較兩個實驗在整個空間中每一個地理格點的 `pred_vs_truth` 全時間平均 RMSE。")

    combos = get_grid_scatter_combos()

    if not combos:
        st.error("❌ 資料庫中找不到預算好的格點 RMSE 數據，請確認是否已執行預處理腳本。")
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
                    'x': f"{exp_x} 格點 RMSE (mm/hr)",
                    'y': f"{exp_y} 格點 RMSE (mm/hr)"
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

            # 設定下載高解析 PNG
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
