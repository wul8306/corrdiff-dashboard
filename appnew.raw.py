import os
import sqlite3
import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==========================================
# 頁面基本配置
# ==========================================
st.set_page_config(
    page_title="CorrDiff 降尺度實驗評估系統",
    page_icon="🌧️",
    layout="wide"
)

DB_PATH = "corrdiff_allmetrics.db"

BASE_NC_DIR = "../plot/out2"
EXPERIMENT_NC_MAP = {
    "SF2SF": os.path.join(BASE_NC_DIR, "SF2SF/netcdf/output_0_all.nc"),
    "SF2SFslp": os.path.join(BASE_NC_DIR, "SF2SFslp/netcdf/output_0_all.nc"),
    "P2P": os.path.join(BASE_NC_DIR, "P2P25Y/netcdf/output_0_all.nc")
}

GRID_COORDS_NC = "wrf_208x208_grid_coords.nc"

# ==========================================
# CWB 標準雨量色階定義 (分級與 Hex 色碼)
# ==========================================
#CWB_PRECIP_LEVELS = [0, 5, 10, 20, 30, 50, 60, 90, 130, 150, 200, 270, 300, 400, 500, 600, 700]
CWB_PRECIP_LEVELS = [0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
CWB_PRECIP_COLORS = [
    "#FFFFFF",  # 0~5: 白色
    "#A0F0FF",  # 5~10: 淺藍
    "#00A0FF",  # 10~20: 天藍
    "#0060FF",  # 20~30: 深藍
    "#0000FF",  # 30~50: 正藍
    "#00A000",  # 50~60: 綠色
    "#00E000",  # 60~90: 亮綠
    "#FFFF00",  # 90~130: 黃色
    "#FFC000",  # 130~150: 橘黃
    "#FF8000",  # 150~200: 橘色
    "#FF0000",  # 200~270: 紅色
    "#D00000",  # 270~300: 深紅
    "#960000",  # 300~400: 暗紅/棕紅
    "#6E006E",  # 400~500: 深紫
    "#B400D2",  # 500~600: 紫色
    "#FF00FF",  # 600~700: 洋紅
    "#FFC0FF"   # >700: 淺粉紫
]

def create_cwb_colorscale():
    """創建符合 Plotly 格式的 CWB 色階 (歸一化至 0~1)"""
    max_val = CWB_PRECIP_LEVELS[-1]
    norm_levels = [v / max_val for v in CWB_PRECIP_LEVELS]
    
    colorscale = []
    for i in range(len(norm_levels) - 1):
        colorscale.append([norm_levels[i], CWB_PRECIP_COLORS[i]])
        colorscale.append([norm_levels[i+1], CWB_PRECIP_COLORS[i]])
    colorscale.append([1.0, CWB_PRECIP_COLORS[-1]])
    return colorscale

CWB_COLORSCALE = create_cwb_colorscale()

# ==========================================
# 快取載入 wrf_208x208 地形與 Landmask
# ==========================================
@st.cache_data
def load_wrf_grid_coords(coords_path=GRID_COORDS_NC):
    if not os.path.exists(coords_path):
        return None, None, None, None

    ds_coords = xr.open_dataset(coords_path)
    lats = ds_coords['XLAT'].values
    lons = ds_coords['XLONG'].values
    land_mask = ds_coords['LANDMASK'].values
    ter = ds_coords['TER'].values
    ds_coords.close()

    return lats, lons, land_mask, ter

# ==========================================
# 資料庫快取載入
# ==========================================
@st.cache_data
def load_all_time_metrics(db_path=DB_PATH):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    query = """
    SELECT exp_name, file_name, comparison_type, variable_name, 
           rmse, fss_threshold, fss_window_size, fss_score
    FROM experiment_metrics
    WHERE comparison_type = 'pred_vs_truth'
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data
def get_grid_scatter_combos(db_path=DB_PATH):
    if not os.path.exists(db_path):
        return []
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
    if not os.path.exists(db_path):
        return None, None, None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ratio_y_gt_x, x_values, y_values 
    FROM grid_rmse_scatter 
    WHERE exp_x=? AND exp_y=? AND var_name=?
    """, (exp_x, exp_y, var_name))
    row = cursor.fetchone()
    
    is_swapped = False
    if not row:
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

@st.cache_data
def get_nc_time_list(nc_path):
    if not os.path.exists(nc_path):
        return None
    ds = xr.open_dataset(nc_path)
    try:
        if np.issubdtype(ds['time'].dtype, np.datetime64):
            time_series = pd.to_datetime(ds['time'].values)
        else:
            base_date = pd.Timestamp("2023-01-01 00:00:00")
            time_series = [base_date + pd.Timedelta(days=float(t)) for t in ds['time'].values]
        times = [t.strftime('%Y-%m-%d %H:%M') for t in time_series]
    except Exception:
        times = [f"Step {idx} (Day Offset: {t})" for idx, t in enumerate(ds['time'].values)]

    ds.close()
    return times

@st.cache_data
def load_spatial_slice(nc_path, time_idx, var_name):
    ds_truth = xr.open_dataset(nc_path, group="truth")
    ds_pred = xr.open_dataset(nc_path, group="prediction")
    ds_input = xr.open_dataset(nc_path, group="input")

    truth_2d = ds_truth[var_name].isel(time=time_idx).values
    input_2d = ds_input[var_name].isel(time=time_idx).values

    pred_var = ds_pred[var_name].isel(time=time_idx)
    if "ensemble" in pred_var.dims:
        pred_2d = pred_var.mean(dim="ensemble").values
    else:
        pred_2d = pred_var.values

    ds_truth.close()
    ds_pred.close()
    ds_input.close()

    return truth_2d, pred_2d, input_2d

# ==========================================
# 側邊欄
# ==========================================
st.sidebar.title("🌧️ CorrDiff 評估儀表板")
tab_selection = st.sidebar.radio(
    "請選擇分析模式：",
    [
        "📊 全時間平均指標 (RMSE & FSS)", 
        "🗺️ 跨實驗格點 RMSE 散佈圖",
        "🌍 空間場數據繪圖 (Spatial Plot)"
    ]
)

# ==========================================
# TAB 1: 指標頁面
# ==========================================
if tab_selection == "📊 全時間平均指標 (RMSE & FSS)":
    st.title("📊 各實驗全時間平均指標 (Prediction vs. Truth)")
    
    df_metrics = load_all_time_metrics(DB_PATH)
    if df_metrics is None:
        st.error(f"❌ 找不到資料庫檔案 `{DB_PATH}`，請確認先執行了預處理腳本！")
        st.stop()
    elif df_metrics.empty:
        st.warning("⚠️ 資料庫中尚無指標資料。")
        st.stop()

    all_exps = sorted(df_metrics['exp_name'].unique().tolist())
    selected_exps = st.sidebar.multiselect("選擇要比較的實驗：", options=all_exps, default=all_exps)

    if not selected_exps:
        st.info("請至少選擇一個實驗。")
        st.stop()

    df_filtered = df_metrics[df_metrics['exp_name'].isin(selected_exps)]

    st.subheader("1. 跨實驗總平均 RMSE 對比")
    df_rmse_avg = df_filtered.groupby('exp_name', as_index=False)['rmse'].mean().sort_values('rmse')

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

    fig_rmse = px.bar(
        df_rmse_avg, x='exp_name', y='rmse', color='exp_name', text_auto='.4f',
        title="Prediction vs. Truth 全時間平均 RMSE (mm/hr)"
    )
    fig_rmse.update_traces(textposition='outside')
    fig_rmse.update_layout(showlegend=False, yaxis_range=[0, df_rmse_avg['rmse'].max() * 1.25])
    st.plotly_chart(fig_rmse, use_container_width=True)

# ==========================================
# TAB 2: 格點 Scatter 頁面
# ==========================================
elif tab_selection == "🗺️ 跨實驗格點 RMSE 散佈圖":
    st.title("🗺️ 跨實驗格點 RMSE 比較 (Grid-level Scatter)")
    combos = get_grid_scatter_combos(DB_PATH)

    if not combos:
        st.error("❌ 資料庫中找不到格點 RMSE 數據。")
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
        ratio_y_gt_x, x_draw, y_draw = load_grid_scatter_data(exp_x, exp_y, var_name="precipitation", db_path=DB_PATH)
        if x_draw is not None:
            ratio_x_gt_y = 100.0 - ratio_y_gt_x
            fig_scatter = px.scatter(
                x=x_draw, y=y_draw, opacity=0.35,
                labels={'x': f"{exp_x} 格點 RMSE", 'y': f"{exp_y} 格點 RMSE"},
                title=f"【Pred vs. Truth 格點 RMSE 對比】{exp_x} (X軸) vs. {exp_y} (Y軸)"
            )
            max_val = max(float(x_draw.max()), float(y_draw.max())) * 1.05
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Red", width=2, dash="dash"))
            fig_scatter.update_xaxes(range=[0, max_val])
            fig_scatter.update_yaxes(range=[0, max_val])
            
            annotation_text = f"<b>Y > X (Y軸較差) 比例: {ratio_y_gt_x:.1f}%</b><br>X > Y (X軸較差) 比例: {ratio_x_gt_y:.1f}%"
            fig_scatter.add_annotation(
                xref="paper", yref="paper", x=0.03, y=0.95, text=annotation_text,
                showarrow=False, align="left", font=dict(size=14, color="black"),
                bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="gray", borderwidth=1, borderpad=6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 3: 空間場數據繪圖 (套用 CWB 雨量色階)
# ==========================================
elif tab_selection == "🌍 空間場數據繪圖 (Spatial Plot)":
    st.title("🌍 空間場數據繪圖 (CWB雨量標準色階)")

    lats, lons, land_mask, ter = load_wrf_grid_coords(GRID_COORDS_NC)
    if lats is None:
        st.error(f"❌ 找不到座標檔案 `{GRID_COORDS_NC}`，請確認檔案位置。")
        st.stop()

    exp_list = list(EXPERIMENT_NC_MAP.keys())
    selected_exp = st.sidebar.selectbox("選擇實驗組別：", options=exp_list, index=0)
    nc_path = EXPERIMENT_NC_MAP[selected_exp]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ 海岸線圖層控制")
    show_boundary = st.sidebar.checkbox("顯示海岸線輪廓", value=True)
    boundary_color = st.sidebar.color_picker("海岸線顏色", "#000000")
    boundary_width = st.sidebar.slider("海岸線粗細", min_value=1, max_value=5, value=2)

    if not os.path.exists(nc_path):
        st.error(f"❌ 找不到 NetCDF 檔案：`{nc_path}`")
        st.stop()

    times = get_nc_time_list(nc_path)
    if times is None:
        st.error("無法讀取 NC 檔案的時間座標。")
        st.stop()

    col_var, col_time = st.columns([1, 2])
    with col_var:
        var_options = {
            "降雨量 (precipitation)": "precipitation",
            "2米氣溫 (temperature_2m)": "temperature_2m",
            "10米東向風 (eastward_wind_10m)": "eastward_wind_10m",
            "10米北向風 (northward_wind_10m)": "northward_wind_10m"
        }
        selected_var_label = st.selectbox("選擇氣象變數：", options=list(var_options.keys()))
        var_name = var_options[selected_var_label]

    with col_time:
        selected_time_str = st.selectbox("選擇事件日期/時間：", options=times, index=0)
        time_idx = times.index(selected_time_str)

    truth_2d, pred_2d, input_2d = load_spatial_slice(nc_path, time_idx, var_name)
    diff_2d = pred_2d - truth_2d

    x_coords = lons[0, :] if lons.ndim == 2 else lons
    y_coords = lats[:, 0] if lats.ndim == 2 else lats

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Ground Truth (真實觀測)",
            f"Prediction (模型預測)",
            f"LR Input (低解析輸入)",
            f"Error Difference (Pred - Truth)"
        ),
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    # 針對降雨量，套用 CWB 色階 (0~700 mm)
    if var_name == "precipitation":
        heatmap_args = dict(
            colorscale=CWB_COLORSCALE,
            zmin=0, zmax=700,
            colorbar=dict(
                tickvals=CWB_PRECIP_LEVELS,
                ticktext=[str(v) for v in CWB_PRECIP_LEVELS]
            )
        )
    else:
        heatmap_args = dict(colorscale="Thermal")

    # 1. Truth
    fig.add_trace(go.Heatmap(z=truth_2d, x=x_coords, y=y_coords, **heatmap_args), row=1, col=1)
    
    # 2. Prediction
    fig.add_trace(go.Heatmap(z=pred_2d, x=x_coords, y=y_coords, **heatmap_args), row=1, col=2)
    
    # 3. Input
    fig.add_trace(go.Heatmap(z=input_2d, x=x_coords, y=y_coords, **heatmap_args), row=2, col=1)
    
    # 4. Difference (維持雙極藍紅圖)
    fig.add_trace(go.Heatmap(z=diff_2d, x=x_coords, y=y_coords, colorscale="RdBu_r", zmid=0), row=2, col=2)

    # 海岸線疊加
    if show_boundary and land_mask is not None:
        boundary_trace = go.Contour(
            z=land_mask,
            x=x_coords,
            y=y_coords,
            contours_type='constraint',
            contours_operation='=',
            contours_value=0.5,
            line=dict(color=boundary_color, width=boundary_width),
            showscale=False,
            hoverinfo='skip'
        )
        fig.add_trace(boundary_trace, row=1, col=1)
        fig.add_trace(boundary_trace, row=1, col=2)
        fig.add_trace(boundary_trace, row=2, col=1)
        fig.add_trace(boundary_trace, row=2, col=2)

    fig.update_xaxes(title_text="經度 Longitude", showgrid=False)
    fig.update_yaxes(title_text="緯度 Latitude", showgrid=False)
    fig.update_layout(height=800, title_text=f"【{selected_exp}】{selected_time_str} 空間對比", margin=dict(l=20, r=20, t=80, b=20))

    st.plotly_chart(fig, use_container_width=True)
