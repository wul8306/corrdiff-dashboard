import json
import os
import sqlite3
import cartopy.crs as ccrs
import geocat.viz.util as gvutil
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap  # 修正：匯入 ListedColormap
import math
import io
import yaml

# ==============================================================================
#   載入 YAML 設定檔 for experiments
# ==============================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到設定檔: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

# ==============================================
# 將 Matplotlib Figure 物件轉為 PNG Byte 流供下載
# ===============================================
def fig_to_bytes(fig, dpi=300):
    buf = io.BytesIO()
    # bbox_inches='tight' 可確保圖例與標題不被切掉
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    return buf

# ==========================================
# 0. 設定 Matplotlib 中文字型
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

plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. 頁面基本配置與 CSS 樣式設定
# ==========================================
st.set_page_config(
    page_title="CorrDiff 降尺度實驗評估系統", page_icon="🌧️", layout="wide"
)

st.markdown(
    """
    <style>
    ul[role="listbox"]::-webkit-scrollbar { width: 18px !important; }
    ul[role="listbox"]::-webkit-scrollbar-track { background: #f1f1f1 !important; border-radius: 8px !important; }
    ul[role="listbox"]::-webkit-scrollbar-thumb { background: #888 !important; border-radius: 8px !important; }
    ul[role="listbox"]::-webkit-scrollbar-thumb:hover { background: #555 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#DB_PATH = os.path.join(CURRENT_DIR, "corrdiff_allmetrics_sea.db")
DB_PATH = os.path.join(CURRENT_DIR, "corrdiff_allmetrics_sea_land.db")
CONFIG_PATH = "config.yaml"
config = load_config(CONFIG_PATH)
EXPERIMENT_NC_MAP = config["experiments"]
for exp_name, nc_path in EXPERIMENT_NC_MAP.items():
    print(f"  • {exp_name} -> {nc_path}")

GRID_COORDS_NC = "wrf_208x208_grid_coords.nc"


# ==========================================
# 2. CWB 標準雨量色階定義 (修正為 ListedColormap)
# ==========================================
PRECIP_BOUNDS = [0, 5, 10, 20, 30, 50, 60, 90, 130, 150, 200, 270, 300, 400, 500, 600, 700]
PRECIP_BOUNDS_meiyu = [0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
PRECIP_COLORS = [
    "#FFFFFF", "#A0F0FF", "#00A0FF", "#0060FF", "#0000FF", "#00A000", "#00E000", "#FFFF00",
    "#FFC000", "#FF8000", "#FF0000", "#D00000", "#960000", "#6E006E", "#B400D2", "#FF00FF", "#FFC0FF"
]

# 正確做法：採用 ListedColormap 將顏色直接與 BoundaryNorm 區間一對一綁定
cwb_cmap = ListedColormap(PRECIP_COLORS)
cwb_norm = BoundaryNorm(PRECIP_BOUNDS, ncolors=cwb_cmap.N, extend="max")
cwb_norm_meiyu = BoundaryNorm(PRECIP_BOUNDS_meiyu, ncolors=cwb_cmap.N, extend="max")

# ==========================================
# 3. 快取載入與資料庫讀取函式
# ==========================================
@st.cache_data
def load_wrf_grid_coords(coords_path=GRID_COORDS_NC):
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


def load_scatter_data(table_name, exp_x, exp_y, var_name="precipitation", db_path=DB_PATH):
    if not os.path.exists(db_path):
        return None, None, None, None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT valid_time, ratio_y_gt_x, x_values, y_values FROM {table_name} "
            "WHERE exp_x=? AND exp_y=? AND var_name=?",
            (exp_x, exp_y, var_name),
        )
        row = cursor.fetchone()
        is_swapped = False
        if not row:
            cursor.execute(
                f"SELECT valid_time, ratio_y_gt_x, x_values, y_values FROM {table_name} "
                "WHERE exp_x=? AND exp_y=? AND var_name=?",
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
        valid_time = tuple(json.loads(raw_valid_time))
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
def load_spatial_slice(nc_path: str, time_idx: int, var_name: str, only_pred: bool = False):
    try:
        # 1. 讀取 Prediction Group (所有實驗都需要)
        with xr.open_dataset(nc_path, group="prediction") as ds_pred:
            da_pred = ds_pred[var_name]
            if "ensemble" in da_pred.dims:
                da_pred = da_pred.mean(dim="ensemble")
            p_2d = da_pred.isel(time=time_idx).values

        # 🔹 如果只需要 prediction，直接在此 return，不用花時間開 truth 與 input group
        if only_pred:
            return None, p_2d, None

        # 2. 讀取 Truth Group (僅第一筆實驗需要)
        with xr.open_dataset(nc_path, group="truth") as ds_truth:
            t_2d = ds_truth[var_name].isel(time=time_idx).values

        # 3. 讀取 Input Group (僅第一筆實驗需要，若無 input group 則給 None)
        try:
            with xr.open_dataset(nc_path, group="input") as ds_input:
                i_2d = ds_input[var_name].isel(time=time_idx).values
        except Exception:
            i_2d = None

        return t_2d, p_2d, i_2d

    except Exception:
        return None, None, None

@st.cache_data
def load_spatial_slice_old(nc_path, time_idx, var_name):
    with (
        xr.open_dataset(nc_path, group="truth") as ds_truth,
        xr.open_dataset(nc_path, group="prediction") as ds_pred,
        xr.open_dataset(nc_path, group="input") as ds_input,
    ):
        truth_2d = ds_truth[var_name].isel(time=time_idx).values
        input_2d = ds_input[var_name].isel(time=time_idx).values
        pred_var = ds_pred[var_name].isel(time=time_idx)
        pred_2d = (
            pred_var.mean(dim="ensemble").values
            if "ensemble" in pred_var.dims
            else pred_var.values
        )
    return truth_2d, pred_2d, input_2d


# ==========================================
# 4. 繪圖共用模組 (含 TAB 4 重構修復)
# ==========================================
def render_scatter_plot(table_name, mode_name):
    combos = get_scatter_combos(table_name, DB_PATH)
    if not combos:
        st.error(f"❌ 資料庫 `{DB_PATH}` 中找不到 `{table_name}` 的預算資料。")
        return

    all_exps = sorted(list(set([r[0] for r in combos] + [r[1] for r in combos])))
    col_x, col_y = st.columns(2)
    with col_x:
        exp_x = st.selectbox("選擇 X 軸實驗 (Baseline)", all_exps, index=0, key=f"{table_name}_x")
    with col_y:
        exp_y = st.selectbox(
            "選擇 Y 軸實驗 (Comparison)", all_exps, index=1 if len(all_exps) > 1 else 0, key=f"{table_name}_y"
        )

    if exp_x == exp_y:
        st.warning("⚠️ 請選擇不同的實驗進行比較。")
        return

    valid_time, ratio_y_gt_x, x_draw, y_draw = load_scatter_data(
        table_name, exp_x, exp_y, var_name="precipitation", db_path=DB_PATH
    )

    if x_draw is not None and len(x_draw) > 0:
        df_scatter = pd.DataFrame({"x_rmse": x_draw, "y_rmse": y_draw, "date_info": valid_time})
        fig_scatter = px.scatter(
            df_scatter,
            x="x_rmse",
            y="y_rmse",
            custom_data=["date_info"],
            opacity=0.35 if table_name == "grid_rmse_scatter" else 0.65,
            labels={"x_rmse": f"{exp_x} (mm/hr)", "y_rmse": f"{exp_y} (mm/hr)"},
            title=f"【Pred vs. Truth {mode_name} RMSE 對比】{exp_x} vs. {exp_y}",
        )
        fig_scatter.update_traces(
            hovertemplate=(
                "<b>時間/資料點:</b> %{customdata[0]}<br>"
                f"<b>{exp_x} RMSE:</b> %{{x:.4f}} mm/hr<br>"
                f"<b>{exp_y} RMSE:</b> %{{y:.4f}} mm/hr<extra></extra>"
            )
        )
        max_val = max(float(x_draw.max()), float(y_draw.max())) * 1.05
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash")
        )
        fig_scatter.update_xaxes(range=[0, max_val])
        fig_scatter.update_yaxes(range=[0, max_val])
#        fig_scatter.update_xaxes(range=[0, max_val], showline=True, linecolor="black",tickfont=dict(color="black", size=12),title_font=dict(color="black", size=14))
#        fig_scatter.update_yaxes(range=[0, max_val], showline=True, linecolor="black",tickfont=dict(color="black", size=12),title_font=dict(color="black", size=14),)

        anno_text = f"<b>Y > X 比例: {ratio_y_gt_x:.1f}%</b><br>X > Y 比例: {100-ratio_y_gt_x:.1f}%"
        fig_scatter.add_annotation(
            xref="paper", yref="paper", x=0.03, y=0.95, text=anno_text, showarrow=False, align="left",
            bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="gray", borderwidth=1
        )
        config = {
            "toImageButtonOptions": {
            "format": "png",  # 可選 'png', 'svg', 'jpeg', 'webp'
            "filename": f"{table_name}_{exp_x}_{exp_y}",  # 👈 在這裡設定自訂檔名 (不需要寫副檔名)
            "height": 500,
            "width": 800,
            "scale": 1.5,  # 放大倍率，2 代表 2 倍高解析度
            }   
        }
#       st.plotly_chart(fig_scatter)
        st.plotly_chart(fig_scatter,config=config)
    else:
        st.warning("⚠️ 該組實驗對比無有效散佈資料。")


def plot_spatial_maps(
    var_name,
    time_str,
    exp_name,
    lats,
    lons,
    land_mask,
    truth_2d,
    input_2d,
    pred_2d,
    is_typhoon=False,
):
    """單一實驗空間場繪圖函式 (TAB 4) - 改用 Cartopy + pcolormesh 確保色階與網格完全正確"""
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6.5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    prefix = "【颱風日】" if is_typhoon else ""
    fig.suptitle(f"{prefix}{var_name} - {time_str} ({exp_name})", fontsize=18, y=0.96)

    lon_grid, lat_grid = np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats)
    im = None

    for idx, ax in enumerate(axes):
        data = [truth_2d, input_2d, pred_2d][idx]
        title = ["Target (TReAD)", "Input (ERA5)", f"CorrDiff ({exp_name})"][idx]

#       ax.coastlines(linewidth=0.8, color="black")

        if var_name == "precipitation":
            im = ax.pcolormesh(
                lon_grid,
                lat_grid,
                data,
                cmap=cwb_cmap,
                norm=cwb_norm,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
        else:
            im = ax.pcolormesh(
                lon_grid,
                lat_grid,
                data,
                cmap="turbo",
                transform=ccrs.PlateCarree(),
                shading="auto",
            )

        if land_mask is not None:
            ax.contour(
                lon_grid,
                lat_grid,
                land_mask,
                levels=[0.5],
                colors="black",
                linewidths=0.8,
                transform=ccrs.PlateCarree(),
            )

        gvutil.set_axes_limits_and_ticks(
            ax,
            xlim=(lon_grid.min(), lon_grid.max()),
            ylim=(lat_grid.min(), lat_grid.max()),
            xticks=np.arange(np.ceil(lon_grid.min()), lon_grid.max() + 1, 1),
            yticks=np.arange(np.ceil(lat_grid.min()), lat_grid.max() + 1, 1),
        )
        gvutil.add_lat_lon_ticklabels(ax)
        gvutil.set_titles_and_labels(
            ax, lefttitle=title, lefttitlefontsize=13, righttitle="", righttitlefontsize=9
        )

    fig.subplots_adjust(right=0.84, wspace=0.2)
    cbar = fig.colorbar(
        im,
        cax=fig.add_axes([0.86, 0.18, 0.02, 0.65]),
        extend="max" if var_name == "precipitation" else "neither",
    )
    if var_name == "precipitation":
        # 挑選重點刻度，防止數值文字重疊
        display_ticks = [0, 10, 30, 60, 130, 200, 400, 700]
        cbar.set_ticks(display_ticks)
        cbar.set_label("降雨量 (mm/hr)", fontsize=12)

    st.pyplot(fig)
    plt.close(fig)


def plot_multi_exp_spatial_maps(
    exp_data_dict,
    lons,
    lats,
    var_name="Precipitation",
    time_str="",
    land_mask=None,
    is_typhoon=True,
    levels=None,
    cmap=cwb_cmap,
    norm=None,  
):
    """多實驗空間分布圖 (TAB 5)- 支援動態雙列佈局"""
    num_plots = len(exp_data_dict)

    # 核心修改：固定一列最多 3 張圖，列數依據總數自動增加
    max_cols_per_row = 5
    n_cols = min(num_plots, max_cols_per_row)               # 1~3 張時維持實際張數，超過 3 張則固定為 3 欄
    n_rows = math.ceil(num_plots / max_cols_per_row)        # 1~3張: 1列 | 4~6張: 2列 | 7~9張: 3列


    if var_name == "precipitation" or var_name == "Precipitation":
        if levels is None:
            levels = PRECIP_BOUNDS
        if norm is None:
            norm = cwb_norm
    elif levels is None:
        levels = np.linspace(0, 100, 11)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows + 0.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    if num_plots == 1:
        axes = [axes]

    # 統一將 axes 轉為 1D array 方便迭代
    axes_flat = np.atleast_1d(axes).ravel()

    lon_grid, lat_grid = np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats)
    cf = None

#   for ax, (title_name, data) in zip(axes, exp_data_dict.items()):
#--------------------------------------- mask out
#       plot_data = (
#           np.where(land_mask > 0, data, np.nan) if land_mask is not None else data
#       )
# 繪製有資料的實驗圖
    for ax, (title_name, data) in zip(axes_flat[:num_plots], exp_data_dict.items()):
        plot_data = data
        cf = ax.contourf(
            lon_grid,
            lat_grid,
            plot_data,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            extend="max" if (var_name == "precipitation" or var_name == "Precipitation") else "neither",
        )

        ax.coastlines(resolution='10m', linewidth=1.0, color='black', zorder=3)

        gvutil.set_axes_limits_and_ticks(
            ax,
            xlim=(lon_grid.min(), lon_grid.max()),
            ylim=(lat_grid.min(), lat_grid.max()),
            xticks=np.arange(np.ceil(lon_grid.min()), lon_grid.max() + 1, 1.0),
            yticks=np.arange(np.ceil(lat_grid.min()), lat_grid.max() + 1, 1.0),
        )
        gvutil.add_lat_lon_ticklabels(ax)
        gvutil.set_titles_and_labels(
            ax,
            lefttitle=title_name,
            lefttitlefontsize=13,
            righttitle=f"{var_name}\n{time_str}" if time_str else var_name,
            righttitlefontsize=9,
        )

    # 隱藏多餘的空白子圖 (例如 5 個實驗於 2x3 網格時隱藏第 6 個)
    for ax in axes_flat[num_plots:]:
        ax.set_visible(False)

    # Colorbar 僅參照實際有繪圖的 axes
    cbar = fig.colorbar(
        cf, ax=axes_flat[:num_plots], orientation="horizontal", shrink=0.5, pad=0.08, aspect=30
#       cf, ax=axes, orientation="horizontal", shrink=0.5, pad=0.08, aspect=30
    )
    cbar.set_label(f"{var_name} Level", fontsize=11)

    return fig, axes


# ==========================================
# 5. 側邊欄選單與頁面邏輯
# ==========================================
st.sidebar.title("🌧️ CorrDiff 評估儀表板")
tab_selection = st.sidebar.radio(
    "請選擇校驗模組：",
    [
        "📊 全時間平均指標 (RMSE & FSS)",
        "📅 季節性 RMSE 比較 (Seasonal)",
        "🗺️ 跨實驗 RMSE 散佈圖比較",
        "🌍 空間場數據繪圖 (Spatial Plot)",
        "🌀 颱風日空間場繪圖 (Typhoon Plot)",
        "🌧️ 梅雨季極端雨量分析 (Meiyu Plot)",
    ],
)
st.sidebar.markdown("---")

if tab_selection == "📊 全時間平均指標 (RMSE & FSS)":
    st.title("📊 各實驗全時間平均指標 (Prediction vs. Truth)")
    df_metrics = load_all_time_metrics(DB_PATH)
    if df_metrics is None or df_metrics.empty:
        st.error(f"❌ 找不到資料庫檔案 `{DB_PATH}` 或無資料，請先執行預處理腳本。")
        st.stop()

    all_exps = sorted(df_metrics["exp_name"].unique().tolist())
    selected_exps = st.sidebar.multiselect("選擇要比較的實驗：", options=all_exps, default=all_exps)
    if not selected_exps:
        st.info("請至少選擇一個實驗。")
        st.stop()

    df_filtered = df_metrics[df_metrics["exp_name"].isin(selected_exps)]
    st.subheader("1. 跨實驗總平均 RMSE 對比 (越低越好)")
    df_rmse_avg = df_filtered.groupby("exp_name", as_index=False)["rmse"].mean().sort_values("rmse")
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
        df_rmse_avg, x="exp_name", y="rmse", color="exp_name", text_auto=".4f",
        title="Prediction vs. Truth 全時間平均 RMSE (mm/hr)",
        labels={"exp_name": "實驗名稱", "rmse": "平均 RMSE (mm/hr)"},
    )
    fig_rmse.update_traces(textposition="outside")
    fig_rmse.update_layout(showlegend=False, yaxis_range=[0, df_rmse_avg["rmse"].max() * 1.25])
    st.plotly_chart(fig_rmse)

    st.markdown("---")
    st.subheader("2. Fractional Skill Score (FSS) 平均表現 (越高越好)")
    fss_tab1, fss_tab2, fss_tab3 = st.tabs(["🎯 特定門檻/視窗", "📈 FSS 隨門檻變化曲線", "🔥 FSS 矩陣熱力圖"])

    with fss_tab1:
        c1, c2 = st.columns(2)
        with c1:
            selected_th = st.selectbox("選擇降雨門檻 (mm/hr):", sorted(df_filtered["fss_threshold"].unique()), index=2)
        with c2:
            selected_w = st.selectbox("選擇視窗大小:", sorted(df_filtered["fss_window_size"].unique()), index=2)

        df_fss_sub = (
            df_filtered[
                (df_filtered["fss_threshold"] == selected_th)
                & (df_filtered["fss_window_size"] == selected_w)
            ]
            .groupby("exp_name", as_index=False)["fss_score"]
            .mean()
        )
        fig_fss_bar = px.bar(df_fss_sub, x="exp_name", y="fss_score", color="exp_name", text_auto=".3f")
        fig_fss_bar.update_layout(showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_fss_bar)

    with fss_tab2:
        selected_w_tab2 = st.selectbox(
            "選擇空間視窗大小:", sorted(df_filtered["fss_window_size"].unique()), index=2, key="w_tab2"
        )
        df_fss_line = (
            df_filtered[df_filtered["fss_window_size"] == selected_w_tab2]
            .groupby(["exp_name", "fss_threshold"], as_index=False)["fss_score"]
            .mean()
        )
        fig_fss_line = px.line(
            df_fss_line, x="fss_threshold", y="fss_score", color="exp_name", markers=True,
            labels={"fss_threshold": "降雨門檻 Threshold (mm/hr)", "fss_score": "平均 FSS 分數"},
        )
        fig_fss_line.update_layout(yaxis_range=[0, 1.05])
        fig_fss_line.update_xaxes(type="category")
        st.plotly_chart(fig_fss_line)

    with fss_tab3:
        exp_for_hm = st.selectbox("選擇實驗：", selected_exps, key="hm_exp")
        df_hm = (
            df_filtered[df_filtered["exp_name"] == exp_for_hm]
            .groupby(["fss_threshold", "fss_window_size"], as_index=False)["fss_score"]
            .mean()
        )
        pivot_fss = df_hm.pivot(index="fss_window_size", columns="fss_threshold", values="fss_score")
        fig_hm = px.imshow(pivot_fss, text_auto=".3f", color_continuous_scale="Viridis", zmin=0, zmax=1)
        fig_hm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_hm)

elif tab_selection == "📅 季節性 RMSE 比較 (Seasonal)":
    st.title("📅 各實驗不同季節的 RMSE 比較")
    df_seasonal = load_seasonal_metrics(DB_PATH)
    if df_seasonal is None or df_seasonal.empty:
        st.error(f"❌ 找不到資料庫檔案 `{DB_PATH}` 或 seasonal_rmse 表內無資料。")
        st.stop()

    all_exps = sorted(df_seasonal["exp_name"].unique().tolist())
    selected_exps = st.sidebar.multiselect("選擇要比較的實驗：", options=all_exps, default=all_exps, key="seasonal_exps")
    if not selected_exps:
        st.info("請至少從左側選擇一個實驗進行比較。")
        st.stop()

    df_filtered_season = df_seasonal[
        (df_seasonal["exp_name"].isin(selected_exps)) & (df_seasonal["variable_name"] == "precipitation")
    ]
    st.subheader("不同季節的平均 RMSE (mm/hr)")
    season_order = ["DJF", "MA", "MJ", "JAS", "ON"]

    fig_season = px.bar(
        df_filtered_season, x="season", y="rmse_mean", color="exp_name", barmode="group", text_auto=".4f",
        category_orders={"season": season_order},
        labels={"season": "季節 (Season)", "rmse_mean": "平均 RMSE (mm/hr)", "exp_name": "實驗名稱"},
        title="各實驗在不同季節的降雨 RMSE 表現 (數值越低越好)",
    )
    # 2. 設定右上角下載按鈕的參數
    config = {
        "toImageButtonOptions": {
            "format": "png",  # 可選 'png', 'svg', 'jpeg', 'webp'
            "filename": f"rmse_seabar",  # 👈 在這裡設定自訂檔名 (不需要寫副檔名)
            "height": 500,
            "width": 800,
            "scale": 2,  # 放大倍率，2 代表 2 倍高解析度
        }
    }   
    fig_season.update_traces(textposition="outside")
    fig_season.update_layout(yaxis_range=[0, df_filtered_season["rmse_mean"].max() * 1.25])
#   st.plotly_chart(fig_season)
    st.plotly_chart(fig_season, config=config)

elif tab_selection == "🗺️ 跨實驗 RMSE 散佈圖比較":
    st.title("🗺️ 跨實驗 RMSE 散佈圖比較 (Scatter Plot)")
    sub_tab_grid, sub_tab_time = st.tabs(["🗺️ 格點層級散佈圖 (grid_rmse_scatter)", "⏱️ 時間層級散佈圖 (time_rmse_scatter)"])
    with sub_tab_grid:
        render_scatter_plot("grid_rmse_scatter", "格點層級 (Grid)")
    with sub_tab_time:
        render_scatter_plot("time_rmse_scatter", "時間層級 (Time)")

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
        var_opts = {"降雨量 (precipitation)": "precipitation", "2米氣溫": "temperature_2m"}
        var_name = var_opts[st.selectbox("選擇變數：", list(var_opts.keys()))]
    with col_time:
        selected_time_str = st.selectbox("選擇日期/時間：", times, index=0)
        time_idx = times.index(selected_time_str)

    truth_2d, pred_2d, input_2d = load_spatial_slice_old(nc_path, time_idx, var_name)
    plot_spatial_maps(
        var_name, selected_time_str, selected_exp, lats, lons, land_mask, truth_2d, input_2d, pred_2d, is_typhoon=False
    )

elif tab_selection == "🌀 颱風日空間場繪圖 (Typhoon Plot)":
    st.title("🌀 颱風日跨實驗空間場對比 (Target / Input / All Experiments)")
    lats, lons, land_mask, ter = load_wrf_grid_coords(GRID_COORDS_NC)
    if lats is None:
        st.error(f"❌ 找不到座標檔案 `{GRID_COORDS_NC}`，請確認檔案位置。")
        st.stop()

    TY_FILE = "2016_2023_Tydays.txt"
    ty_dates = load_typhoon_dates(TY_FILE)
    if not ty_dates:
        st.error(f"❌ 找不到檔案 `{TY_FILE}` 或檔案無效。")
        st.stop()

    ref_nc_path = next((p for p in EXPERIMENT_NC_MAP.values() if os.path.exists(p)), None)
    if not ref_nc_path:
        st.error("❌ 在 `EXPERIMENT_NC_MAP` 中找不到任何有效的 NC 檔案。")
        st.stop()

    all_times = get_nc_time_list(ref_nc_path)
    col_var, col_date, col_time = st.columns([1, 2, 1])
    with col_var:
        var_opts = {"降雨量 (precipitation)": "precipitation", "2米氣溫": "temperature_2m"}
        var_name = var_opts[st.selectbox("選擇變數：", list(var_opts.keys()), key="ty_var")]
    with col_date:
        selected_ty_date = st.selectbox("選擇颱風日期 (YYYY-MM-DD)：", ty_dates, index=0)

    available_times = [t for t in all_times if t.startswith(selected_ty_date)]
    if not available_times:
        st.warning(f"⚠️ 找不到日期 `{selected_ty_date}` 的有效時間紀錄。")
        st.stop()

    with col_time:
        selected_time_str = st.selectbox("選擇時間點：", available_times, index=0, key="ty_time")
        time_idx = all_times.index(selected_time_str)

    truth_2d, input_2d = None, None
    exp_preds_dict, missing_exps = {}, []

    for exp_k, path_v in EXPERIMENT_NC_MAP.items():
        if os.path.exists(path_v):
            try:
                if truth_2d is None:
                    # 1. 第一筆成功讀取的實驗：同步保留 truth_2d 與 input_2d
                    truth_2d, p_2d, input_2d = load_spatial_slice(path_v, time_idx, var_name, only_pred=False)
                else:
                    # 2. 後續實驗：僅保留 p_2d，忽略 truth 與 input (以 _ 接收)
                    _, p_2d, _ = load_spatial_slice(path_v, time_idx, var_name, only_pred=True)

                exp_preds_dict[exp_k] = p_2d
            except Exception:
                missing_exps.append(exp_k)
        else:
            missing_exps.append(exp_k)

    if missing_exps:
        st.info(f"ℹ️ 提示：未找到以下實驗檔，將跳過繪製：{', '.join(missing_exps)}")

    if not exp_preds_dict or truth_2d is None:
        st.error("❌ 無法載入任何實驗的繪圖數據。")
        st.stop()

    all_maps_dict = {"Target(TReAD)": truth_2d, "Input(ERA5)": input_2d, **exp_preds_dict}
    fig, axes = plot_multi_exp_spatial_maps(
        exp_data_dict=all_maps_dict,
        lons=lons,
        lats=lats,
        var_name=var_name,
        time_str=selected_time_str,
        land_mask=land_mask,
    )
    # 2. 將圖像轉為 Byte 資料
    img_bytes = fig_to_bytes(fig, dpi=300)

    # 3. 使用 columns 控制右上角下載按鈕版面
    col_left, col_btn = st.columns([0.75, 0.25])

    with col_btn:
    # 放置於右上角的下載按鈕
        st.download_button(
            label="📥 下載空間雨量圖 (PNG)",
            data=img_bytes,
            file_name=f"typhoon_spatial_map_{var_name}_{selected_ty_date}.png",
            mime="image/png",
            use_container_width=True,  # 讓按鈕自動填滿欄位寬度
            key=f"dl_tab5",  # 加上獨特 key 避免跨 Tab 衝突
        )
    st.pyplot(fig)
    plt.close(fig)

elif tab_selection == "🌧️ 梅雨季極端雨量分析 (Meiyu Plot)":
    st.title("🌧️梅雨季極端雨量跨實驗空間場對比 (Target / Input / All Experiments)")
    lats, lons, land_mask, ter = load_wrf_grid_coords(GRID_COORDS_NC)
    if lats is None:
        st.error(f"❌ 找不到座標檔案 `{GRID_COORDS_NC}`，請確認檔案位置。")
        st.stop()

    TY_FILE = "2016_2023_MJ_extremed.txt"
    ty_dates = load_typhoon_dates(TY_FILE)
    if not ty_dates:
        st.error(f"❌ 找不到檔案 `{TY_FILE}` 或檔案無效。")
        st.stop()

    ref_nc_path = next((p for p in EXPERIMENT_NC_MAP.values() if os.path.exists(p)), None)
    if not ref_nc_path:
        st.error("❌ 在 `EXPERIMENT_NC_MAP` 中找不到任何有效的 NC 檔案。")
        st.stop()

    all_times = get_nc_time_list(ref_nc_path)
    col_var, col_date, col_time = st.columns([1, 2, 1])
    with col_var:
        var_opts = {"降雨量 (precipitation)": "precipitation", "2米氣溫": "temperature_2m"}
        var_name = var_opts[st.selectbox("選擇變數：", list(var_opts.keys()), key="ty_var")]
    with col_date:
        selected_ty_date = st.selectbox("選擇日期 (YYYY-MM-DD)：", ty_dates, index=0)

    available_times = [t for t in all_times if t.startswith(selected_ty_date)]
    if not available_times:
        st.warning(f"⚠️ 找不到日期 `{selected_ty_date}` 的有效時間紀錄。")
        st.stop()

    with col_time:
        selected_time_str = st.selectbox("選擇時間點：", available_times, index=0, key="ty_time")
        time_idx = all_times.index(selected_time_str)

    truth_2d, input_2d = None, None
    exp_preds_dict, missing_exps = {}, []

    for exp_k, path_v in EXPERIMENT_NC_MAP.items():
        if os.path.exists(path_v):
            try:
                if truth_2d is None:
                    # 1. 第一筆成功讀取的實驗：同步保留 truth_2d 與 input_2d
                    truth_2d, p_2d, input_2d = load_spatial_slice(path_v, time_idx, var_name, only_pred=False)
                else:
                    # 2. 後續實驗：僅保留 p_2d，忽略 truth 與 input (以 _ 接收)
                    _, p_2d, _ = load_spatial_slice(path_v, time_idx, var_name, only_pred=True)

                exp_preds_dict[exp_k] = p_2d
            except Exception:
                missing_exps.append(exp_k)
        else:
            missing_exps.append(exp_k)

    if missing_exps:
        st.info(f"ℹ️ 提示：未找到以下實驗檔，將跳過繪製：{', '.join(missing_exps)}")

    if not exp_preds_dict or truth_2d is None:
        st.error("❌ 無法載入任何實驗的繪圖數據。")
        st.stop()

    all_maps_dict = {"Target(TReAD)": truth_2d, "Input(ERA5)": input_2d, **exp_preds_dict}
    fig, axes = plot_multi_exp_spatial_maps(
        exp_data_dict=all_maps_dict,
        lons=lons,
        lats=lats,
        levels=PRECIP_BOUNDS_meiyu,
        norm=cwb_norm_meiyu,
        var_name=var_name,
        time_str=selected_time_str,
        land_mask=land_mask,
    )

    # 2. 將圖像轉為 Byte 資料
    img_bytes = fig_to_bytes(fig, dpi=300)

    # 3. 使用 columns 控制右上角下載按鈕版面
    col_left, col_btn = st.columns([0.75, 0.25])

    with col_btn:
    # 放置於右上角的下載按鈕
        st.download_button(
            label="📥 下載空間雨量圖 (PNG)",
            data=img_bytes,
            file_name=f"meiyu_spatial_map_{var_name}_{selected_ty_date}.png",
            mime="image/png",
            use_container_width=True,  # 讓按鈕自動填滿欄位寬度
            key=f"dl_tab6",  # 加上獨特 key 避免跨 Tab 衝突
        )

    st.pyplot(fig)
    plt.close(fig)
