import json
import os
import sqlite3
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

# ==========================================
# 0. 設定 Matplotlib 中文字型 (無需 sudo 權限)
# ==========================================
FONT_PATH = "font.ttf"

if os.path.exists(FONT_PATH):
    my_font = fm.FontProperties(fname=FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.sans-serif"] = [my_font.get_name()]
else:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "Microsoft JhengHei",
        "DejaVu Sans",
    ]

plt.rcParams["axes.unicode_minus"] = False  # 避免負號顯示異常


# ==========================================
# 1. 頁面基本配置與 CSS 樣式設定
# ==========================================
st.set_page_config(
    page_title="CorrDiff 降尺度實驗評估系統", page_icon="🌧️", layout="wide"
)

# 自訂 CSS：加寬所有下拉選單 (st.selectbox) 出現時的滾動條 (Scrollbar)
st.markdown(
    """
    <style>
    /* 針對下拉選單清單 (role="listbox") 的捲軸進行加寬 */
    ul[role="listbox"]::-webkit-scrollbar {
        width: 18px !important; /* 捲軸寬度，數字越大越寬 (預設約 6px) */
    }
    
    ul[role="listbox"]::-webkit-scrollbar-track {
        background: #f1f1f1 !important; /* 捲軸軌道背景色 */
        border-radius: 8px !important;
    }
    
    ul[role="listbox"]::-webkit-scrollbar-thumb {
        background: #888 !important; /* 捲軸主體顏色 */
        border-radius: 8px !important;
    }
    
    ul[role="listbox"]::-webkit-scrollbar-thumb:hover {
        background: #555 !important; /* 滑鼠懸停時的顏色 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "corrdiff_allmetrics_sea.db")
BASE_NC_DIR = "output"
EXPERIMENT_NC_MAP = {
    "P2P": os.path.join(BASE_NC_DIR, "P2P25Y/output_0_all.nc"),
    "CP2P": os.path.join(BASE_NC_DIR, "CP2P25Y/output_0_all.nc"),
}

# WRF 208x208 網格與地形座標檔
GRID_COORDS_NC = "wrf_208x208_grid_coords.nc"


# ==========================================
# 2. CWB 標準雨量色階與 BoundaryNorm 定義
# ==========================================
PRECIP_BOUNDS = [0, 5, 10, 20, 30, 50, 60, 90, 130, 150, 200, 270, 300, 400, 500, 600, 700]
PRECIP_COLORS = [
    "#FFFFFF", "#A0F0FF", "#00A0FF", "#0060FF", "#0000FF", "#00A000", "#00E000", "#FFFF00",
    "#FFC000", "#FF8000", "#FF0000", "#D00000", "#960000", "#6E006E", "#B400D2", "#FF00FF", "#FFC0FF"
]
cwb_cmap = LinearSegmentedColormap.from_list(
    "cwb_precip_cmap", PRECIP_COLORS, N=len(PRECIP_COLORS)
)
cwb_norm = BoundaryNorm(PRECIP_BOUNDS, ncolors=cwb_cmap.N, extend="max")


# ==========================================
# 3. 快取載入與資料庫讀取函式
# ==========================================
@st.cache_data
def load_wrf_grid_coords(coords_path=GRID_COORDS_NC):
    """載入專屬 WRF 網格座標 (XLAT, XLONG, LANDMASK, TER)"""
    if not os.path.exists(coords_path):
        return None, None, None, None
    with xr.open_dataset(coords_path) as ds_coords:
        lats = ds_coords["XLAT"].values
        lons = ds_coords["XLONG"].values
        land_mask = ds_coords["LANDMASK"].values
        ter = ds_coords["TER"].values
    return lats, lons, land_mask, ter


@st.cache_data
def load_all_time_metrics(db_path=DB_PATH):
    """載入所有實驗的全時間平均指標"""
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    query = """
        SELECT exp_name, file_name, comparison_type, variable_name,
               rmse, fss_threshold, fss_window_size, fss_score
        FROM summary_metrics
        WHERE comparison_type = 'pred_vs_truth'
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


@st.cache_data
def load_seasonal_metrics(db_path=DB_PATH):
    """載入所有實驗的季節性平均 RMSE 指標"""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    query = "SELECT exp_name, season, variable_name, rmse_mean FROM seasonal_rmse"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


@st.cache_data
def load_typhoon_dates(file_path="2016_2023_Tydays.txt"):
    """讀取颱風日期並轉換為 YYYY-MM-DD 格式，方便後續篩選"""
    if not os.path.exists(file_path):
        return []
    dates = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            d = line.strip()
            if len(d) == 8 and d.isdigit():
                dates.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
            elif len(d) > 0:
                dates.append(d)
    return sorted(list(set(dates)))


@st.cache_data
def get_scatter_combos(table_name="grid_rmse_scatter", db_path=DB_PATH):
    """動態讀取指定 scatter 表格中的實驗組合"""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT DISTINCT exp_x, exp_y FROM {table_name}")
        rows = cursor.fetchall()
        return rows
    except Exception:
        return []
    finally:
        conn.close()


def load_scatter_data(
    table_name, exp_x, exp_y, var_name="precipitation", db_path=DB_PATH
):
    """從指定的 scatter 資料表讀取兩個實驗的散佈數據"""
    if not os.path.exists(db_path):
        return None, None, None, None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            f"SELECT valid_time, ratio_y_gt_x, x_values, y_values FROM {table_name}"
            " WHERE exp_x=? AND exp_y=? AND var_name=?",
            (exp_x, exp_y, var_name),
        )
        row = cursor.fetchone()

        is_swapped = False
        if not row:
            cursor.execute(
                f"SELECT valid_time, ratio_y_gt_x, x_values, y_values FROM"
                f" {table_name} WHERE exp_x=? AND exp_y=? AND var_name=?",
                (exp_y, exp_x, var_name),
            )
            row = cursor.fetchone()
            is_swapped = True
    except Exception:
        row = None
    finally:
        conn.close()

    if not row:
        return None, None, None, None

    raw_valid_time = row[0]
    try:
        valid_time = json.loads(raw_valid_time)
        valid_time = tuple(valid_time)
    except (json.JSONDecodeError, TypeError):
        valid_time = raw_valid_time

    ratio_orig = row[1]
    x_draw = np.frombuffer(row[2], dtype=np.float32)
    y_draw = np.frombuffer(row[3], dtype=np.float32)

    if is_swapped:
        x_draw, y_draw = y_draw, x_draw
        ratio_y_gt_x = 100.0 - ratio_orig
    else:
        ratio_y_gt_x = ratio_orig

    return valid_time, ratio_y_gt_x, x_draw, y_draw


@st.cache_data
def get_nc_time_list(nc_path):
    """解析相對時間天數並轉為 YYYY-MM-DD HH:MM (用於 TAB 3 & 4)"""
    if not os.path.exists(nc_path):
        return []
    with xr.open_dataset(nc_path) as ds:
        try:
            if np.issubdtype(ds["time"].dtype, np.datetime64):
                time_series = pd.to_datetime(ds["time"].values)
            else:
                base_date = pd.Timestamp("2016-01-01 00:00:00")
                time_series = [
                    base_date + pd.Timedelta(days=float(t)) for t in ds["time"].values
                ]
            times = [t.strftime("%Y-%m-%d %H:%M") for t in time_series]
        except Exception:
            times = [f"Step {idx}" for idx in range(len(ds["time"].values))]
    return times


@st.cache_data
def load_spatial_slice(nc_path, time_idx, var_name):
    """讀取 Truth, Prediction (Ensemble平均), Input 的 2D 矩陣"""
    with (
        xr.open_dataset(nc_path, group="truth") as ds_truth,
        xr.open_dataset(nc_path, group="prediction") as ds_pred,
        xr.open_dataset(nc_path, group="input") as ds_input,
    ):
        truth_2d = ds_truth[var_name].isel(time=time_idx).values
        input_2d = ds_input[var_name].isel(time=time_idx).values

        pred_var = ds_pred[var_name].isel(time=time_idx)
        if "ensemble" in pred_var.dims:
            pred_2d = pred_var.mean(dim="ensemble").values
        else:
            pred_2d = pred_var.values

    return truth_2d, pred_2d, input_2d


# ==========================================
# 4. 繪圖共用模組
# ==========================================
def render_scatter_plot(table_name, mode_name):
    """通用散佈圖繪製函式 (TAB 2)"""
    combos = get_scatter_combos(table_name, DB_PATH)
    if not combos:
        st.error(f"❌ 資料庫 `{DB_PATH}` 中找不到 `{table_name}` 的預算資料。")
        return

    all_exps = sorted(list(set([r[0] for r in combos] + [r[1] for r in combos])))

    col_x, col_y = st.columns(2)
    with col_x:
        exp_x = st.selectbox(
            f"選擇 X 軸實驗 (Baseline)",
            all_exps,
            index=0,
            key=f"{table_name}_x",
        )
    with col_y:
        exp_y = st.selectbox(
            f"選擇 Y 軸實驗 (Comparison)",
            all_exps,
            index=1 if len(all_exps) > 1 else 0,
            key=f"{table_name}_y",
        )

    if exp_x == exp_y:
        st.warning("⚠️ 請選擇不同的實驗進行比較。")
        return

    valid_time, ratio_y_gt_x, x_draw, y_draw = load_scatter_data(
        table_name, exp_x, exp_y, var_name="precipitation", db_path=DB_PATH
    )

    if x_draw is not None and len(x_draw) > 0:
        date_info = valid_time
        df_scatter = pd.DataFrame(
            {"x_rmse": x_draw, "y_rmse": y_draw, "date_info": date_info}
        )

        fig_scatter = px.scatter(
            df_scatter,
            x="x_rmse",
            y="y_rmse",
            custom_data=["date_info"],
            opacity=0.35 if table_name == "grid_rmse_scatter" else 0.65,
            labels={
                "x_rmse": f"{exp_x} (mm/hr)",
                "y_rmse": f"{exp_y} (mm/hr)",
            },
            title=f"【Pred vs. Truth {mode_name} RMSE 對比】{exp_x} vs. {exp_y}",
        )

        fig_scatter.update_traces(
            hovertemplate=(
                "<b>時間/資料點:</b> %{customdata[0]}<br>"
                f"<b>{exp_x} RMSE:</b> %{{x:.4f}} mm/hr<br>"
                f"<b>{exp_y} RMSE:</b> %{{y:.4f}} mm/hr"
                "<extra></extra>"
            )
        )

        max_val = max(float(x_draw.max()), float(y_draw.max())) * 1.05
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_val,
            y1=max_val,
            line=dict(color="Red", dash="dash"),
        )
        fig_scatter.update_xaxes(range=[0, max_val])
        fig_scatter.update_yaxes(range=[0, max_val])

        anno_text = (
            f"<b>Y > X 比例: {ratio_y_gt_x:.1f}%</b><br>"
            f"X > Y 比例: {100-ratio_y_gt_x:.1f}%"
        )
        fig_scatter.add_annotation(
            xref="paper",
            yref="paper",
            x=0.03,
            y=0.95,
            text=anno_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="gray",
            borderwidth=1,
        )
        st.plotly_chart(fig_scatter)
    else:
        st.warning("⚠️ 該組實驗對比無有效散佈資料。")


def plot_spatial_maps(var_name, time_str, exp_name, lats, lons, land_mask, truth_2d, input_2d, pred_2d, is_typhoon=False):
    """單一實驗空間場繪圖函式 (TAB 3)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    prefix = "【颱風日】" if is_typhoon else ""
    fig.suptitle(
        f"{prefix}{var_name} - {time_str} ({exp_name})", fontsize=18, y=0.96
    )

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    for idx, ax in enumerate(axes):
        data = [truth_2d, input_2d, pred_2d][idx]
        title = ["Target (TReAD)", "Input (ERA5)", f"CorrDiff ({exp_name})"][idx]

        im = ax.imshow(
            data,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=cwb_cmap if var_name == "precipitation" else "turbo",
            norm=cwb_norm if var_name == "precipitation" else None,
        )

        if land_mask is not None:
            ax.contour(
                lons, lats, land_mask, levels=[0.5], colors="black", linewidths=1.0
            )

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("經度 (Longitude)")
        if idx == 0:
            ax.set_ylabel("緯度 (Latitude)")

    fig.subplots_adjust(right=0.84, wspace=0.2)
    cbar = fig.colorbar(
        im,
        cax=fig.add_axes([0.86, 0.18, 0.02, 0.65]),
        extend="max" if var_name == "precipitation" else "neither",
    )
    if var_name == "precipitation":
        cbar.set_ticks(PRECIP_BOUNDS)
        cbar.set_label("降雨量 (mm/hr)", fontsize=12)

    st.pyplot(fig)
    plt.close(fig)


def plot_multi_exp_spatial_maps(var_name, time_str, lats, lons, land_mask, truth_2d, input_2d, exp_preds_dict, is_typhoon=True):
    """多實驗同圖繪製函式 (TAB 4)"""
    num_exps = len(exp_preds_dict)
    num_panels = 2 + num_exps  # Target, Input + 各實驗預測
    
    # 動態調整子圖寬度
    fig, axes = plt.subplots(1, num_panels, figsize=(5.2 * num_panels, 6.5))
    if num_panels == 1:
        axes = [axes]

    prefix = "【颱風日多實驗對比】" if is_typhoon else "【多實驗對比】"
    fig.suptitle(f"{prefix}{var_name} - {time_str}", fontsize=18, y=0.96)

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    panel_datas = [truth_2d, input_2d] + list(exp_preds_dict.values())
    panel_titles = ["Target (TReAD)", "Input (ERA5)"] + [f"CorrDiff ({exp_k})" for exp_k in exp_preds_dict.keys()]

    for idx, ax in enumerate(axes):
        data = panel_datas[idx]
        title = panel_titles[idx]

        im = ax.imshow(
            data,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=cwb_cmap if var_name == "precipitation" else "turbo",
            norm=cwb_norm if var_name == "precipitation" else None,
        )

        if land_mask is not None:
            ax.contour(
                lons, lats, land_mask, levels=[0.5], colors="black", linewidths=1.0
            )

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("經度 (Longitude)")
        if idx == 0:
            ax.set_ylabel("緯度 (Latitude)")

    # Colorbar 放置於最右側
    cbar_x = 0.90 + 0.01 * (4 / max(num_panels, 1))
    fig.subplots_adjust(right=0.88, wspace=0.22)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        extend="max" if var_name == "precipitation" else "neither",
    )
    if var_name == "precipitation":
        cbar.set_ticks(PRECIP_BOUNDS)
        cbar.set_label("降雨量 (mm/hr)", fontsize=12)

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 5. 側邊欄選單
# ==========================================
st.sidebar.title("🌧️ CorrDiff 評估儀表板")
tab_selection = st.sidebar.radio(
    "請選擇分析模式：",
    [
        "📊 全時間平均指標 (RMSE & FSS)",
        "📅 季節性 RMSE 比較 (Seasonal)",
        "🗺️ 跨實驗 RMSE 散佈圖比較",
        "🌍 空間場數據繪圖 (Spatial Plot)",
        "🌀 颱風日空間場繪圖 (Typhoon Plot)",
    ],
)
st.sidebar.markdown("---")


# ==========================================
# TAB 1: 全時間平均指標分析 (RMSE & FSS)
# ==========================================
if tab_selection == "📊 全時間平均指標 (RMSE & FSS)":
    st.title("📊 各實驗全時間平均指標 (Prediction vs. Truth)")

    df_metrics = load_all_time_metrics(DB_PATH)
    if df_metrics is None or df_metrics.empty:
        st.error(
            f"❌ 找不到資料庫檔案 `{DB_PATH}` 或無資料，請先執行預處理腳本。"
        )
        st.stop()

    all_exps = sorted(df_metrics["exp_name"].unique().tolist())
    selected_exps = st.sidebar.multiselect(
        "選擇要比較的實驗：", options=all_exps, default=all_exps
    )

    if not selected_exps:
        st.info("請至少選擇一個實驗。")
        st.stop()

    df_filtered = df_metrics[df_metrics["exp_name"].isin(selected_exps)]

    # 1. 總平均 RMSE
    st.subheader("1. 跨實驗總平均 RMSE 對比 (越低越好)")
    df_rmse_avg = (
        df_filtered.groupby("exp_name", as_index=False)["rmse"]
        .mean()
        .sort_values("rmse")
    )
    cols = st.columns(len(df_rmse_avg))
    best_rmse = df_rmse_avg["rmse"].min()

    for idx, row in df_rmse_avg.reset_index(drop=True).iterrows():
        with cols[idx]:
            if row["rmse"] == best_rmse:
                st.metric(f"🏆 {row['exp_name']} (最佳)", f"{row['rmse']:.4f} mm/hr")
            else:
                st.metric(
                    row["exp_name"],
                    f"{row['rmse']:.4f} mm/hr",
                    delta=f"+{row['rmse'] - best_rmse:.4f}",
                    delta_color="inverse",
                )

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
    fss_tab1, fss_tab2, fss_tab3 = st.tabs(
        ["🎯 特定門檻/視窗", "📈 FSS 隨門檻變化曲線", "🔥 FSS 矩陣熱力圖"]
    )

    with fss_tab1:
        c1, c2 = st.columns(2)
        with c1:
            selected_th = st.selectbox(
                "選擇降雨門檻 (mm/hr):",
                sorted(df_filtered["fss_threshold"].unique()),
                index=2,
            )
        with c2:
            selected_w = st.selectbox(
                "選擇視窗大小:",
                sorted(df_filtered["fss_window_size"].unique()),
                index=2,
            )

        df_fss_sub = (
            df_filtered[
                (df_filtered["fss_threshold"] == selected_th)
                & (df_filtered["fss_window_size"] == selected_w)
            ]
            .groupby("exp_name", as_index=False)["fss_score"]
            .mean()
        )
        fig_fss_bar = px.bar(
            df_fss_sub,
            x="exp_name",
            y="fss_score",
            color="exp_name",
            text_auto=".3f",
        )
        fig_fss_bar.update_layout(showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_fss_bar)

    with fss_tab2:
        selected_w_tab2 = st.selectbox(
            "選擇空間視窗大小:",
            sorted(df_filtered["fss_window_size"].unique()),
            index=2,
            key="w_tab2",
        )
        df_fss_line = (
            df_filtered[df_filtered["fss_window_size"] == selected_w_tab2]
            .groupby(["exp_name", "fss_threshold"], as_index=False)["fss_score"]
            .mean()
        )

        fig_fss_line = px.line(
            df_fss_line,
            x="fss_threshold",
            y="fss_score",
            color="exp_name",
            markers=True,
            labels={
                "fss_threshold": "降雨門檻 Threshold (mm/hr)",
                "fss_score": "平均 FSS 分數",
            },
        )
        fig_fss_line.update_layout(yaxis_range=[0, 1.05])
        fig_fss_line.update_xaxes(type="category")
        st.plotly_chart(fig_fss_line)

    with fss_tab3:
        exp_for_hm = st.selectbox("選擇實驗：", selected_exps, key="hm_exp")
        df_hm = (
            df_filtered[df_filtered["exp_name"] == exp_for_hm]
            .groupby(["fss_threshold", "fss_window_size"], as_index=False)[
                "fss_score"
            ]
            .mean()
        )
        pivot_fss = df_hm.pivot(
            index="fss_window_size", columns="fss_threshold", values="fss_score"
        )

        fig_hm = px.imshow(
            pivot_fss, text_auto=".3f", color_continuous_scale="Viridis", zmin=0, zmax=1
        )
        fig_hm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_hm)


# ==========================================
# TAB 2: 季節性 RMSE 比較
# ==========================================
elif tab_selection == "📅 季節性 RMSE 比較 (Seasonal)":
    st.title("📅 各實驗不同季節的 RMSE 比較")

    df_seasonal = load_seasonal_metrics(DB_PATH)
    if df_seasonal is None or df_seasonal.empty:
        st.error(f"❌ 找不到資料庫檔案 `{DB_PATH}` 或 seasonal_rmse 表內無資料，請確認已執行最新版的資料前處理。")
        st.stop()

    all_exps = sorted(df_seasonal["exp_name"].unique().tolist())
    selected_exps = st.sidebar.multiselect(
        "選擇要比較的實驗：", options=all_exps, default=all_exps, key="seasonal_exps"
    )

    if not selected_exps:
        st.info("請至少從左側選擇一個實驗進行比較。")
        st.stop()

    df_filtered_season = df_seasonal[
        (df_seasonal["exp_name"].isin(selected_exps)) &
        (df_seasonal["variable_name"] == "precipitation")
    ]

    st.subheader("不同季節的平均 RMSE (mm/hr)")
    season_order = ["DJF", "MA", "MJ", "JAS", "ON"]

    fig_season = px.bar(
        df_filtered_season,
        x="season",
        y="rmse_mean",
        color="exp_name",
        barmode="group",
        text_auto=".4f",
        category_orders={"season": season_order},
        labels={
            "season": "季節 (Season)",
            "rmse_mean": "平均 RMSE (mm/hr)",
            "exp_name": "實驗名稱"
        },
        title="各實驗在不同季節的降雨 RMSE 表現 (數值越低越好)"
    )

    fig_season.update_traces(textposition="outside")
    max_rmse = df_filtered_season["rmse_mean"].max()
    fig_season.update_layout(yaxis_range=[0, max_rmse * 1.25])
    st.plotly_chart(fig_season)


# ==========================================
# TAB 3: 跨實驗 RMSE 散佈圖比較
# ==========================================
elif tab_selection == "🗺️ 跨實驗 RMSE 散佈圖比較":
    st.title("🗺️ 跨實驗 RMSE 散佈圖比較 (Scatter Plot)")

    sub_tab_grid, sub_tab_time = st.tabs([
        "🗺️ 格點層級散佈圖 (grid_rmse_scatter)",
        "⏱️ 時間層級散佈圖 (time_rmse_scatter)",
    ])

    with sub_tab_grid:
        render_scatter_plot("grid_rmse_scatter", "格點層級 (Grid)")

    with sub_tab_time:
        render_scatter_plot("time_rmse_scatter", "時間層級 (Time)")


# ==========================================
# TAB 4: 空間場數據繪圖 (Spatial Plot - 單一實驗)
# ==========================================
elif tab_selection == "🌍 空間場數據繪圖 (Spatial Plot)":
    st.title("🌍 空間場數據繪圖 (Target / Input / Prediction 對比)")

    lats, lons, land_mask, ter = load_wrf_grid_coords(GRID_COORDS_NC)
    if lats is None:
        st.error(f"❌ 找不到座標檔案 `{GRID_COORDS_NC}`，請確認檔案位置。")
        st.stop()

    exp_list = list(EXPERIMENT_NC_MAP.keys())
    selected_exp = st.sidebar.selectbox("選擇實驗組別：", options=exp_list, index=0)
    nc_path = EXPERIMENT_NC_MAP[selected_exp]

    if not os.path.exists(nc_path):
        st.error(f"❌ 找不到檔案：`{nc_path}`\n請檢查相對路徑！")
        st.stop()

    times = get_nc_time_list(nc_path)

    col_var, col_time = st.columns([1, 2])
    with col_var:
        var_opts = {
            "降雨量 (precipitation)": "precipitation",
            "2米氣溫": "temperature_2m",
        }
        var_name = var_opts[st.selectbox("選擇變數：", list(var_opts.keys()))]
    with col_time:
        selected_time_str = st.selectbox("選擇日期/時間：", times, index=0)
        time_idx = times.index(selected_time_str)

    truth_2d, pred_2d, input_2d = load_spatial_slice(nc_path, time_idx, var_name)

    plot_spatial_maps(
        var_name, selected_time_str, selected_exp, 
        lats, lons, land_mask, truth_2d, input_2d, pred_2d, is_typhoon=False
    )


# ==========================================
# TAB 5: 颱風日空間場繪圖 (Typhoon Plot - 所有實驗跨實驗同圖比較)
# ==========================================
elif tab_selection == "🌀 颱風日空間場繪圖 (Typhoon Plot)":
    st.title("🌀 颱風日跨實驗空間場對比 (Target / Input / All Experiments)")

    # 1. 載入網格座標
    lats, lons, land_mask, ter = load_wrf_grid_coords(GRID_COORDS_NC)
    if lats is None:
        st.error(f"❌ 找不到座標檔案 `{GRID_COORDS_NC}`，請確認檔案位置。")
        st.stop()

    # 2. 載入颱風日 TXT 清單
    TY_FILE = "2016_2023_Tydays.txt"
    ty_dates = load_typhoon_dates(TY_FILE)
    if not ty_dates:
        st.error(f"❌ 找不到檔案 `{TY_FILE}` 或檔案無效，請確認檔案是否有資料且放置於正確資料夾。")
        st.stop()

    # 3. 尋找可用的 NC 檔案以取得時間清單
    ref_nc_path = None
    for exp_k, path_v in EXPERIMENT_NC_MAP.items():
        if os.path.exists(path_v):
            ref_nc_path = path_v
            break

    if not ref_nc_path:
        st.error(f"❌ 在 `EXPERIMENT_NC_MAP` 中找不到任何有效的 NC 檔案，請檢查檔案路徑。")
        st.stop()

    all_times = get_nc_time_list(ref_nc_path)

    col_var, col_date, col_time = st.columns([1, 2, 1])
    with col_var:
        var_opts = {
            "降雨量 (precipitation)": "precipitation",
            "2米氣溫": "temperature_2m",
        }
        var_name = var_opts[st.selectbox("選擇變數：", list(var_opts.keys()), key="ty_var")]

    # 4. 下拉選單選擇目標日期
    with col_date:
        selected_ty_date = st.selectbox("選擇颱風日期 (YYYY-MM-DD)：", ty_dates, index=0)

    # 5. 過濾找出所選日期中「有包含的資料時間點」
    available_times = [t for t in all_times if t.startswith(selected_ty_date)]

    if not available_times:
        st.warning(f"⚠️ 在資料庫中，找不到日期 `{selected_ty_date}` 的有效時間紀錄。")
        st.stop()

    # 6. 下拉選單選擇時間點
    with col_time:
        selected_time_str = st.selectbox("選擇時間點：", available_times, index=0, key="ty_time")
        time_idx = all_times.index(selected_time_str)

    # 7. 載入所有實驗的數據 (Target 與 Input 取自參考檔案，Pred 取自各實驗)
    truth_2d = None
    input_2d = None
    exp_preds_dict = {}
    missing_exps = []

    for exp_k, path_v in EXPERIMENT_NC_MAP.items():
        if os.path.exists(path_v):
            try:
                t_2d, p_2d, i_2d = load_spatial_slice(path_v, time_idx, var_name)
                if truth_2d is None:
                    truth_2d = t_2d
                    input_2d = i_2d
                exp_preds_dict[exp_k] = p_2d
            except Exception as e:
                missing_exps.append(exp_k)
        else:
            missing_exps.append(exp_k)

    if missing_exps:
        st.info(f"ℹ️ 提示：未找到以下實驗檔，將跳過繪製：{', '.join(missing_exps)}")

    if not exp_preds_dict or truth_2d is None:
        st.error("❌ 無法載入任何實驗的繪圖數據。")
        st.stop()

    # 8. 繪製包含 Target, Input 與所有實驗預測結果的同圖比較
    plot_multi_exp_spatial_maps(
        var_name, selected_time_str,
        lats, lons, land_mask, truth_2d, input_2d, exp_preds_dict, is_typhoon=True
    )

