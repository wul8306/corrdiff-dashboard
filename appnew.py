import os
import sqlite3
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. 頁面基本配置
# ==========================================
st.set_page_config(
    page_title="CorrDiff 降尺度實驗評估系統",
    page_icon="🌧️",
    layout="wide"
)

# DB_PATH = "corrdiff_metrics.db"
# 實驗 NC 檔案路徑設定 (請依實體伺服器路徑微調)
#BASE_NC_DIR = "../plot/out2"
#EXPERIMENT_NC_MAP = {
#    "SF2SF": os.path.join(BASE_NC_DIR, "SF2SF/netcdf/output_0_all.nc"),
#    "SF2SFslp": os.path.join(BASE_NC_DIR, "SF2SFslp/netcdf/output_0_all.nc"),
#    "P2P": os.path.join(BASE_NC_DIR, "P2P25Y/netcdf/output_0_all.nc")
#}
DB_PATH = "corrdiff_allmetrics.db"
BASE_NC_DIR = "output"
EXPERIMENT_NC_MAP = {
    "P2P25Y": os.path.join(BASE_NC_DIR, "P2P25Y/output_0_all.nc"),
    "CP2P25Y": os.path.join(BASE_NC_DIR, "CP2P25Y/output_0_all.nc")
}

# WRF 208x208 網格與地形座標檔
GRID_COORDS_NC = "wrf_208x208_grid_coords.nc"

# ==========================================
# 2. CWB 標準雨量色階與 BoundaryNorm 定義
# ==========================================
PRECIP_BOUNDS = [0, 5, 10, 20, 30, 50, 60, 90, 130, 150, 200, 270, 300, 400, 500, 600, 700]
PRECIP_COLORS = [
    "#FFFFFF",  # 0~5: 白
    "#A0F0FF",  # 5~10: 淺藍
    "#00A0FF",  # 10~20: 天藍
    "#0060FF",  # 20~30: 深藍
    "#0000FF",  # 30~50: 正藍
    "#00A000",  # 50~60: 綠
    "#00E000",  # 60~90: 亮綠
    "#FFFF00",  # 90~130: 黃
    "#FFC000",  # 130~150: 橘黃
    "#FF8000",  # 150~200: 橘
    "#FF0000",  # 200~270: 紅
    "#D00000",  # 270~300: 深紅
    "#960000",  # 300~400: 棕紅
    "#6E006E",  # 400~500: 深紫
    "#B400D2",  # 500~600: 紫
    "#FF00FF",  # 600~700: 洋紅
    "#FFC0FF"   # >700: 淺粉紫
]

# 建立獨立的 Matplotlib Colormap & BoundaryNorm
cwb_cmap = LinearSegmentedColormap.from_list("cwb_precip_cmap", PRECIP_COLORS, N=len(PRECIP_COLORS))
cwb_norm = BoundaryNorm(PRECIP_BOUNDS, ncolors=cwb_cmap.N, extend='max')

# ==========================================
# 3. 快取載入與資料庫讀取函式
# ==========================================
@st.cache_data
def load_wrf_grid_coords(coords_path=GRID_COORDS_NC):
    """載入專屬 WRF 網格座標 (XLAT, XLONG, LANDMASK, TER)"""
    if not os.path.exists(coords_path):
        return None, None, None, None

    ds_coords = xr.open_dataset(coords_path)
    lats = ds_coords['XLAT'].values
    lons = ds_coords['XLONG'].values
    land_mask = ds_coords['LANDMASK'].values  # 1: 陸地, 0: 海洋
    ter = ds_coords['TER'].values
    ds_coords.close()

    return lats, lons, land_mask, ter

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
    """解析相對時間天數並轉為 YYYY-MM-DD HH:MM"""
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
    """讀取 Truth, Prediction (Ensemble平均), Input 的 2D 矩陣"""
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
# 4. 側邊欄選單
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

st.sidebar.markdown("---")


# ==========================================
# TAB 1: 全時間平均指標分析 (RMSE & FSS)
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

    # 1. 總平均 RMSE 卡片與長條圖
    st.subheader("1. 跨實驗總平均 RMSE 對比 (越低越好)")
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
        title="Prediction vs. Truth 全時間平均 RMSE (mm/hr)",
        labels={'exp_name': '實驗名稱', 'rmse': '平均 RMSE (mm/hr)'}
    )
    fig_rmse.update_traces(textposition='outside')
    fig_rmse.update_layout(showlegend=False, yaxis_range=[0, df_rmse_avg['rmse'].max() * 1.25])
    st.plotly_chart(fig_rmse, width="stretch")

    st.markdown("---")

    # 2. FSS 分數圖表與熱力圖
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
            df_fss_sub, x='exp_name', y='fss_score', color='exp_name', text_auto='.3f',
            title=f"平均 FSS 分數 (門檻: {selected_th} mm/hr, 視窗: {selected_w}x{selected_w})",
            labels={'exp_name': '實驗名稱', 'fss_score': 'FSS Score'}
        )
        fig_fss_bar.update_traces(textposition='outside')
        fig_fss_bar.update_layout(showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_fss_bar, width="stretch")

    with fss_tab2:
        st.markdown("##### 選擇實驗觀看「視窗大小 vs 降雨門檻」的全域 FSS 分佈矩陣：")
        exp_for_hm = st.selectbox("選擇實驗：", options=selected_exps, key="hm_exp")
        
        df_hm = df_filtered[df_filtered['exp_name'] == exp_for_hm].groupby(
            ['fss_threshold', 'fss_window_size'], as_index=False
        )['fss_score'].mean()

        pivot_fss = df_hm.pivot(index='fss_window_size', columns='fss_threshold', values='fss_score')

        fig_hm = px.imshow(
            pivot_fss, text_auto='.3f', color_continuous_scale='Viridis', zmin=0, zmax=1,
            labels=dict(x="降雨門檻 Threshold (mm/hr)", y="視窗大小 Window Size", color="FSS"),
            title=f"【{exp_for_hm}】不同門檻與視窗下之平均 FSS 熱力圖"
        )
        fig_hm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_hm, width="stretch")


# ==========================================
# TAB 2: 跨實驗格點 RMSE 散佈圖比較
# ==========================================
elif tab_selection == "🗺️ 跨實驗格點 RMSE 散佈圖":
    st.title("🗺️ 跨實驗格點 RMSE 比較 (Grid-level Scatter)")
    st.caption("比較兩個實驗在整個空間中每一個地理格點的全時間平均 RMSE。")

    combos = get_grid_scatter_combos(DB_PATH)

    if not combos:
        st.error("❌ 資料庫中找不到預算好的格點 RMSE 數據。")
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
                labels={'x': f"{exp_x} 格點 RMSE (mm/hr)", 'y': f"{exp_y} 格點 RMSE (mm/hr)"},
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
            st.plotly_chart(fig_scatter, width="stretch")


# ==========================================
# TAB 3: 空間場數據繪圖 (Matplotlib + CWB Colormap + WRF 海岸線)
# ==========================================
elif tab_selection == "🌍 空間場數據繪圖 (Spatial Plot)":
    st.title("🌍 空間場數據繪圖 (Target / Input / Prediction 對比)")

    # 1. 載入網格座標與 Landmask
    lats, lons, land_mask, ter = load_wrf_grid_coords(GRID_COORDS_NC)
    if lats is None:
        st.error(f"❌ 找不到座標檔案 `{GRID_COORDS_NC}`，請確認檔案位置。")
        st.stop()

    exp_list = list(EXPERIMENT_NC_MAP.keys())
    selected_exp = st.sidebar.selectbox("選擇實驗組別：", options=exp_list, index=0)
    nc_path = EXPERIMENT_NC_MAP[selected_exp]

    if not os.path.exists(nc_path):
        st.error(f"❌ 找不到 NetCDF 檔案：`{nc_path}`\n請檢查 `EXPERIMENT_NC_MAP` 相對路徑！")
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

    # 讀取 2D 矩陣切片
    truth_precip, pred_precip, input_precip = load_spatial_slice(nc_path, time_idx, var_name)

    # 建立 Matplotlib 1x3 子圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle(f"{selected_var_label} - {selected_time_str} ({selected_exp})", fontsize=18, y=0.96)

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    extent = [lon_min, lon_max, lat_min, lat_max]

    sub_titles = ["Target (TReAD)", "Input (ERA5)", "CorrDiff (Reg)"]
    data_list = [truth_precip, input_precip, pred_precip]

    for idx, ax in enumerate(axes):
        if var_name == "precipitation":
            im = ax.imshow(
                data_list[idx], 
                extent=extent, 
                origin='lower', 
                cmap=cwb_cmap, 
                norm=cwb_norm,
                aspect='equal'
            )
        else:
            im = ax.imshow(
                data_list[idx], 
                extent=extent, 
                origin='lower', 
                cmap='turbo',
                aspect='equal'
            )
        
        if land_mask is not None:
            ax.contour(
                lons, lats, land_mask, 
                levels=[0.5], 
                colors='black', 
                linewidths=1.0
            )

        ax.set_title(sub_titles[idx], fontsize=14, pad=10)
        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

    fig.subplots_adjust(right=0.84, wspace=0.2)
    cbar_ax = fig.add_axes([0.86, 0.18, 0.02, 0.65])
    
    if var_name == "precipitation":
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=PRECIP_BOUNDS, extend='max')
        cbar.set_label('Precipitation (mm/hr)', fontsize=12)
    else:
        cbar = fig.colorbar(im, cax=cbar_ax)

    # 於 Streamlit 渲染 Matplotlib 視窗
    st.pyplot(fig)
